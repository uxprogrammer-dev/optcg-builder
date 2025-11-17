import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { spawn } from 'child_process';
import { delimiter, normalize, resolve, dirname, join } from 'path';
import { existsSync } from 'fs';
import { DeckCardSuggestion, GenerateDeckRequest, LeaderSuggestionWithCard } from '@shared/contracts/ai';
import { OptcgApiService } from '../optcg/optcg-api.service';
import { OptcgCard } from '@shared/types/optcg-card';
import type { DeckSuggestionResult } from './ai.service';
import { LeaderKnowledgeService } from './leader-knowledge.service';
import { CardKnowledgeService } from './card-knowledge.service';

interface MlModelConfig {
  enabled: boolean;
  pythonPath: string;
  module: string;
  modelPath: string;
  promptVocabularyPath: string;
  cardVocabularyPath: string;
  dataRoot: string;
  decodeStrategy: 'beam' | 'greedy';
  beamWidth: number;
  lengthPenalty: number;
  timeoutMs: number;
}

interface MlDeckPayload {
  leader_id: string;
  main_deck: string[];
  sideboard?: string[];
}

@Injectable()
export class MlDeckService {
  private readonly logger = new Logger(MlDeckService.name);
  private readonly config: MlModelConfig;

  constructor(
    private readonly configService: ConfigService,
    private readonly optcgApiService: OptcgApiService,
    private readonly leaderKnowledgeService: LeaderKnowledgeService,
    private readonly cardKnowledgeService: CardKnowledgeService,
  ) {
    // Sanitize module name when loading from config to prevent corruption
    const rawModule = this.configService.get<string>('mlModel.module') ?? 'ml.inference.generate_deck';
    let moduleName = rawModule.trim();
    // Remove any environment variable patterns (KEY=VALUE) that might be appended
    moduleName = moduleName.split(/\s+/)[0].split(/[=;]/)[0];
    // Also remove any trailing non-alphanumeric characters except dots and underscores
    moduleName = moduleName.replace(/[^a-zA-Z0-9._-]+$/, '');
    
    if (moduleName !== rawModule.trim()) {
      this.logger.warn(`Module name was sanitized during config load: "${rawModule}" -> "${moduleName}"`);
    }
    
    this.config = {
      enabled: this.configService.get<boolean>('mlModel.enabled') ?? false,
      pythonPath: this.configService.get<string>('mlModel.pythonPath') ?? 'python',
      module: moduleName,
      modelPath: this.configService.get<string>('mlModel.modelPath') ?? '',
      promptVocabularyPath: this.configService.get<string>('mlModel.promptVocabularyPath') ?? '',
      cardVocabularyPath: this.configService.get<string>('mlModel.cardVocabularyPath') ?? '',
      dataRoot: this.configService.get<string>('mlModel.dataRoot') ?? 'data',
      decodeStrategy:
        (this.configService.get<'beam' | 'greedy'>('mlModel.decodeStrategy') as
          | 'beam'
          | 'greedy'
          | undefined) ?? 'beam',
      beamWidth: this.configService.get<number>('mlModel.beamWidth') ?? 5,
      lengthPenalty: this.configService.get<number>('mlModel.lengthPenalty') ?? 0.7,
      timeoutMs: this.configService.get<number>('mlModel.timeoutMs') ?? 15000,
    };

    if (this.isEnabled) {
      this.logger.log(
        `ML Deck Service enabled: model=${this.config.modelPath}, strategy=${this.config.decodeStrategy}`,
      );
    } else {
      this.logger.warn(
        `ML Deck Service disabled: enabled=${this.config.enabled}, modelPath=${this.config.modelPath ? 'set' : 'missing'}, promptVocab=${this.config.promptVocabularyPath ? 'set' : 'missing'}, cardVocab=${this.config.cardVocabularyPath ? 'set' : 'missing'}`,
      );
    }
  }

  get isEnabled(): boolean {
    return (
      this.config.enabled &&
      Boolean(this.config.modelPath) &&
      (Boolean(this.config.promptVocabularyPath) || this.canDeriveVocabPaths())
    );
  }

  /**
   * Derives vocabulary paths from the model path.
   * Model path format: models/run_YYYYMMDD-HHMMSS/deck_transformer.keras
   * Vocab paths: models/run_YYYYMMDD-HHMMSS/vocab/prompt_vocabulary.txt
   *              models/run_YYYYMMDD-HHMMSS/vocab/card_vocabulary.json
   */
  private deriveVocabPaths(modelPath: string): { promptVocab: string; cardVocab: string } {
    const modelDir = dirname(this.normalizePath(modelPath));
    return {
      promptVocab: join(modelDir, 'vocab', 'prompt_vocabulary.txt'),
      cardVocab: join(modelDir, 'vocab', 'card_vocabulary.json'),
    };
  }

  /**
   * Checks if vocab paths can be derived from the model path.
   */
  private canDeriveVocabPaths(): boolean {
    if (!this.config.modelPath) {
      return false;
    }
    const derived = this.deriveVocabPaths(this.config.modelPath);
    return existsSync(derived.promptVocab) && existsSync(derived.cardVocab);
  }

  async generateDeck(
    request: GenerateDeckRequest,
    progressId?: string,
    progressService?: import('./progress.service').ProgressService,
  ): Promise<DeckSuggestionResult | null> {
    if (!this.isEnabled) {
      this.logger.debug('ML deck generation skipped: service not enabled or misconfigured');
      return null;
    }

    this.logger.log(`Attempting ML deck generation for leader ${request.leaderCardId} with prompt: "${request.prompt.substring(0, 50)}..."`);
    try {
      // Use beam search for better quality (trained on 7700+ tournament decks)
      // Fall back to greedy only if beam search times out
      const beamTimeout = Math.max(this.config.timeoutMs * 3, this.config.timeoutMs);
      let payload: MlDeckPayload | null = null;
      try {
        // Try beam search first for better quality
        progressService?.updateProgress(progressId!, 'Generating deck with beam search (trusting model 100%)...', 30);
        // Only restrict leader if one is already selected, otherwise let model choose freely
        const leaderAllowlist = request.leaderCardId ? this.buildLeaderAllowlist(request.leaderCardId) : undefined;
        payload = await this.invokePythonModel(request, 'beam', beamTimeout, false, leaderAllowlist);
      } catch (primaryError) {
        // If beam search fails, fall back to greedy for speed
        this.logger.debug(`Beam search failed, trying greedy decode...`);
        progressService?.updateProgress(progressId!, 'Beam search timed out, using faster greedy search...', 40);
        const leaderAllowlist = request.leaderCardId ? this.buildLeaderAllowlist(request.leaderCardId) : undefined;
        payload = await this.invokePythonModel(
          request,
          'greedy',
          beamTimeout,
          false,
          leaderAllowlist,
        );
      }

      if (!payload?.leader_id || !payload?.main_deck?.length) {
        this.logger.warn('ML deck generation returned an incomplete payload.');
        return null;
      }

      progressService?.updateProgress(progressId!, 'Processing generated deck...', 50);
      // Normalize IDs to canonical format (e.g., OP01-001) and merge variants
      const normalizedLeaderId = this.normalizeCardSetId(payload.leader_id);
      const normalizedMainDeck = payload.main_deck.map((id) => this.normalizeCardSetId(id));

      const deckCounts = this.buildCardCounts(normalizedMainDeck);
      const uniqueCardIds = Array.from(deckCounts.keys());

      progressService?.updateProgress(progressId!, `Loading ${uniqueCardIds.length} cards from database...`, 55);
      const [leaderCard, deckCards] = await Promise.all([
        this.optcgApiService.getCardBySetId(normalizedLeaderId).catch(() => null),
        this.optcgApiService.getCardsBySetIds(uniqueCardIds),
      ]);

      const allowedSetIds = new Set((request.setIds ?? []).map((id) => this.normalizeSetCode(id)));
      const metaOnly = request.metaOnly ?? false;
      const tournamentOnly = request.tournamentOnly ?? false;

      const cardMap = new Map(deckCards.map((card) => [card.id.toUpperCase(), card]));

      // TRUST MODEL 100%: No filtering - return exactly what the model generated
      const suggestions: DeckCardSuggestion[] = uniqueCardIds.map((cardId) => {
        const rawQty = deckCounts.get(cardId) ?? 0;
        const quantity = Math.min(4, rawQty);
        const card = cardMap.get(cardId);
        return {
          cardSetId: cardId,
          quantity,
          role: this.resolveRole(card),
        };
      });
      
      this.logger.debug(
        `ML generated ${suggestions.length} cards (${uniqueCardIds.length} unique) - trusting model 100%, no filtering applied.`,
      );

      suggestions.sort((a, b) => {
        const quantityDelta = (b.quantity ?? 0) - (a.quantity ?? 0);
        if (quantityDelta !== 0) {
          return quantityDelta;
        }
        return a.cardSetId.localeCompare(b.cardSetId);
      });

      const summary = this.buildSummary(leaderCard, request.prompt);

      const cappedTotal = suggestions.reduce((sum, c) => sum + (c.quantity ?? 0), 0);
      this.logger.log(
        `ML deck generation succeeded: generated ${suggestions.length} unique cards (${cappedTotal} total after cap) for leader ${normalizedLeaderId}`,
      );

      return {
        summary,
        cards: suggestions,
        source: 'ml',
        notes: undefined,
        gameplaySummary: `${request.prompt.trim()}`,
      };
    } catch (error) {
      this.logger.error(`ML deck generation failed: ${(error as Error).message}`, error as Error);
      return null;
    }
  }

  /**
   * Normalizes a path for the current platform.
   * Converts Unix-style paths (e.g., /c/dev/...) to Windows paths (e.g., C:\dev\...)
   * on Windows systems.
   */
  private normalizePath(path: string): string {
    if (!path) {
      return path;
    }

    // On Windows, convert Git Bash/Cygwin style paths (/c/...) to Windows paths (C:\...)
    if (process.platform === 'win32' && path.startsWith('/')) {
      // Handle /c/... -> C:\...
      const match = path.match(/^\/([a-z])\/(.*)$/i);
      if (match) {
        const drive = match[1].toUpperCase();
        const rest = match[2].replace(/\//g, '\\');
        return `${drive}:\\${rest}`;
      }
      // If it's an absolute path starting with / but not /c/, resolve it
      return resolve(path.replace(/\//g, '\\'));
    }

    // For relative paths or already normalized paths, use path.normalize
    return normalize(path);
  }

  private async invokePythonModel(request: GenerateDeckRequest, decodeStrategy?: 'beam' | 'greedy', timeoutMs?: number, leaderOnly?: boolean, firstTokenAllowlist?: string[]): Promise<MlDeckPayload> {
    const pythonPath = this.normalizePath(this.config.pythonPath);
    const modelPath = this.normalizePath(this.config.modelPath);
    
    // Always derive vocab paths from model path (they should be in the same run directory)
    // Fall back to explicit config paths only if derived paths don't exist
    let promptVocabPath: string;
    let cardVocabPath: string;
    
    const derived = this.deriveVocabPaths(modelPath);
    const derivedPromptVocab = this.normalizePath(derived.promptVocab);
    const derivedCardVocab = this.normalizePath(derived.cardVocab);
    
    // Check if derived paths exist, otherwise fall back to explicit config
    if (existsSync(derivedPromptVocab) && existsSync(derivedCardVocab)) {
      promptVocabPath = derivedPromptVocab;
      cardVocabPath = derivedCardVocab;
      this.logger.debug(
        `Using derived vocab paths from model: prompt=${promptVocabPath}, card=${cardVocabPath}`,
      );
    } else if (this.config.promptVocabularyPath && this.config.cardVocabularyPath) {
      // Fall back to explicit config paths if derived paths don't exist
      promptVocabPath = this.normalizePath(this.config.promptVocabularyPath);
      cardVocabPath = this.normalizePath(this.config.cardVocabularyPath);
      this.logger.warn(
        `Derived vocab paths not found, using config paths: prompt=${promptVocabPath}, card=${cardVocabPath}`,
      );
    } else {
      // Use derived paths even if they don't exist (let Python script handle the error)
      promptVocabPath = derivedPromptVocab;
      cardVocabPath = derivedCardVocab;
      this.logger.warn(
        `Using derived vocab paths (may not exist): prompt=${promptVocabPath}, card=${cardVocabPath}`,
      );
    }

    // Use provided decode strategy or fall back to config default
    const strategy = decodeStrategy ?? this.config.decodeStrategy;

    // Compose prompt with leader ability (async to fetch leader card)
    const promptText = await this.composePrompt(request, leaderOnly);
    
    // Sanitize module name to prevent environment variables or paths from being appended
    let moduleName = this.config.module.trim();
    // Remove any environment variable patterns (KEY=VALUE) that might be appended
    moduleName = moduleName.split(/\s+/)[0].split(/[=;]/)[0];
    // Also remove any trailing non-alphanumeric characters except dots and underscores
    moduleName = moduleName.replace(/[^a-zA-Z0-9._-]+$/, '');
    
    if (moduleName !== this.config.module.trim()) {
      this.logger.warn(`Module name was sanitized: "${this.config.module}" -> "${moduleName}"`);
    }
    
    const args = [
      '-m',
      moduleName,
      '--prompt',
      promptText,
      '--model',
      modelPath,
      '--prompt-vocab',
      promptVocabPath,
      '--card-vocab',
      cardVocabPath,
      '--decode-strategy',
      strategy,
    ];

    // Only add beam search parameters if using beam search
    if (strategy === 'beam') {
      args.push('--beam-width', String(this.config.beamWidth));
      args.push('--length-penalty', String(this.config.lengthPenalty));
    }
    // Only add leader-id if provided and not empty
    if (request.leaderCardId && request.leaderCardId.trim() !== '') {
      args.push('--leader-id', request.leaderCardId);
    }
    // Fast leader-only flag for suggestion speed
    if (leaderOnly) {
      args.push('--leader-only');
      // Only add allowlist if provided (trust model 100% otherwise)
      if (firstTokenAllowlist && firstTokenAllowlist.length > 0) {
        const leaderIds = firstTokenAllowlist
          .map((id) => this.normalizeCardSetId(id).toUpperCase())
          .filter((id) => id.length > 0);
        if (leaderIds.length > 0) {
          args.push('--first-token-allowlist', leaderIds.join(','));
          this.logger.debug(`Using allowlist with ${leaderIds.length} leaders: ${leaderIds.slice(0, 5).join(', ')}${leaderIds.length > 5 ? '...' : ''}`);
        } else {
          this.logger.debug('Allowlist provided but empty after normalization - trusting model 100%');
        }
      } else {
        this.logger.debug('No allowlist provided - trusting model 100% to generate freely');
      }
    }

    // Verify Python executable exists
    if (!existsSync(pythonPath)) {
      throw new Error(
        `Python executable not found at: ${pythonPath} (normalized from: ${this.config.pythonPath}). Please verify ML_PYTHON_PATH in your .env file.`,
      );
    }

    this.logger.debug(`Executing: ${pythonPath} ${args.join(' ')}`);

    const child = spawn(pythonPath, args, {
      cwd: process.cwd(),
      env: {
        ...process.env,
        PYTHONPATH: process.env.PYTHONPATH
          ? `${process.env.PYTHONPATH}${delimiter}${process.cwd()}`
          : process.cwd(),
      },
    });

    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (chunk) => {
      stdout += chunk.toString();
    });

    child.stderr.on('data', (chunk) => {
      stderr += chunk.toString();
    });

    const timeoutDuration = timeoutMs ?? this.config.timeoutMs;

    return new Promise<MlDeckPayload>((resolve, reject) => {
      const timeout = setTimeout(() => {
        child.kill('SIGKILL');
        reject(new Error(`ML deck generation timed out after ${timeoutDuration}ms`));
      }, timeoutDuration);

      child.on('error', (error: NodeJS.ErrnoException) => {
        clearTimeout(timeout);
        const errorMessage =
          error.code === 'ENOENT'
            ? `Python executable not found at: ${pythonPath}. Please verify ML_PYTHON_PATH in your .env file.`
            : `Failed to spawn Python process: ${error.message}`;
        reject(new Error(errorMessage));
      });

      child.on('close', (code) => {
        clearTimeout(timeout);
        if (code !== 0) {
          this.logger.error(`ML process exited with ${code}: ${stderr}`);
          // Log debug output from stderr if available
          if (stderr && stderr.includes('DEBUG:')) {
            this.logger.debug(`ML debug output: ${stderr}`);
          }
          reject(new Error(stderr || `Process exited with code ${code}`));
          return;
        }

        // Log debug output from stderr even on success (for troubleshooting)
        if (stderr && stderr.includes('DEBUG:')) {
          this.logger.debug(`ML debug output: ${stderr}`);
        }

        try {
          const payload = JSON.parse(stdout.trim() || '{}');
          resolve(payload);
        } catch (error) {
          reject(new Error(`Failed to parse ML deck payload: ${(error as Error).message}`));
        }
      });
    });
  }

  private buildCardCounts(mainDeck: string[]): Map<string, number> {
    const counts = new Map<string, number>();
    for (const rawId of mainDeck) {
      const key = this.normalizeCardSetId(rawId);
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    return counts;
  }

  private buildLeaderAllowlist(leaderId?: string): string[] | undefined {
    if (!leaderId) {
      return undefined;
    }
    const base = leaderId.toUpperCase();
    const variants = ['','_P1','_P2','_P3','_P4','_R1','_R2','_P5'];
    const allowlist = new Set<string>();
    for (const variant of variants) {
      allowlist.add(`${base}${variant}`);
    }
    return Array.from(allowlist);
  }

  private resolveRole(card?: OptcgCard | null): DeckCardSuggestion['role'] {
    if (!card?.type) {
      return 'other';
    }

    const normalized = card.type.toLowerCase();
    if (normalized.includes('character')) {
      return 'character';
    }
    if (normalized.includes('event')) {
      return 'event';
    }
    if (normalized.includes('stage')) {
      return 'stage';
    }
    if (normalized.includes('don')) {
      return 'don';
    }
    if (normalized.includes('counter')) {
      return 'counter';
    }
    return 'other';
  }

  private buildSummary(leaderCard: OptcgCard | null, prompt: string): string {
    const truncatedPrompt = prompt.length > 140 ? `${prompt.slice(0, 137)}...` : prompt;
    const leaderLabel = leaderCard?.name ?? leaderCard?.id ?? 'selected leader';
    return `ML-generated deck for ${leaderLabel} responding to: ${truncatedPrompt}`;
  }

  private async composePrompt(request: GenerateDeckRequest, isLeaderOnly: boolean = false): Promise<string> {
    // If no leader ID is provided, let the model generate one
    if (!request.leaderCardId || request.leaderCardId.trim() === '') {
      const prompt = request.prompt?.trim() || '';
      // Ensure minimum length for validation
      if (prompt.length < 3) {
        return 'Generate a competitive deck';
      }
      
      // For leader-only generation, enhance the prompt to match training format
      // The model was trained on prompts like "Create a competitive deck for {leader_name}"
      // So we format the user's requirements in a similar structure
      if (isLeaderOnly) {
        const lowerPrompt = prompt.toLowerCase();
        
        // If prompt already starts with "create", "build", "design", etc., use as-is
        if (/^(create|build|design|make|construct|suggest|find)/i.test(prompt)) {
          return prompt;
        }
        
        // Try to expand character name references (e.g., "Luffy" -> "Monkey.D.Luffy")
        // This helps the model match short names to full leader names it was trained on
        let enhancedPrompt = prompt;
        const promptMatch = this.leaderKnowledgeService.matchPrompt(prompt);
        if (promptMatch.keywords.size > 0) {
          // Found keyword matches - check if we can expand character names
          // For example, if "luffy" matches, we could add "Monkey.D.Luffy" context
          // But we'll let the keyword-matched allowlist handle this instead
          // The prompt itself should remain user-friendly
        }
        
        // Format to match training: "Create a competitive deck for {leader_name}"
        // Replace "deck" with the user's requirements if present, otherwise append
        if (lowerPrompt.includes('deck')) {
          // User already mentioned "deck", format as: "Create a competitive {user_prompt}"
          return `Create a competitive ${enhancedPrompt}`;
        }
        
        // No "deck" mentioned, format as: "Create a competitive {user_prompt} deck"
        return `Create a competitive ${enhancedPrompt} deck`;
      }
      
      return prompt;
    }
    
    // Get leader card to include ability text (matches training format)
    const leaderCard = await this.optcgApiService.getCardBySetId(request.leaderCardId).catch(() => null);
    const leaderMetadata = this.leaderKnowledgeService.getLeaderMetadata(request.leaderCardId);
    const leaderName = leaderMetadata?.name || request.leaderCardId;
    const leaderAbility = leaderCard?.text || '';
    
    const userPrompt = (request.prompt?.trim() || '').trim();
    
    // Build leader context header (matches training data format)
    const buildPromptWithContext = (basePrompt: string): string => {
      if (leaderAbility) {
        return `Leader: ${leaderName} (${request.leaderCardId})
Leader Ability: ${leaderAbility}

${basePrompt}`;
      }
      return basePrompt;
    };
    
    if (!userPrompt || userPrompt.length < 3) {
      // Fallback if prompt is too short - include leader ability
      return buildPromptWithContext(`Create a competitive deck for ${leaderName}`);
    }
    
    const lowerPrompt = userPrompt.toLowerCase();
    const lowerLeaderName = leaderName.toLowerCase();
    
    // If the user prompt already mentions the leader name, still include ability for context
    if (lowerPrompt.includes(lowerLeaderName) || 
        lowerPrompt.includes(`with ${lowerLeaderName}`) ||
        lowerPrompt.includes(`for ${lowerLeaderName}`)) {
      return buildPromptWithContext(userPrompt);
    }
    
    // Check if prompt already contains "deck" - insert leader name naturally
    if (lowerPrompt.includes('deck')) {
      const deckIndex = lowerPrompt.indexOf('deck');
      const afterDeck = userPrompt.substring(deckIndex + 5).trim();
      const beforeDeck = userPrompt.substring(0, deckIndex + 5);
      
      if (afterDeck) {
        if (afterDeck.toLowerCase().startsWith('for ') || afterDeck.toLowerCase().startsWith('with ')) {
          if (afterDeck.toLowerCase().includes(lowerLeaderName)) {
            return buildPromptWithContext(userPrompt);
          }
          const afterWith = afterDeck.replace(/^(for|with)\s+\w+(\s+that)?/i, '').trim();
          if (afterWith) {
            return buildPromptWithContext(`${beforeDeck} for ${leaderName} that ${afterWith}`.trim());
          }
          return buildPromptWithContext(`${beforeDeck} for ${leaderName}`.trim());
        }
        return buildPromptWithContext(`${beforeDeck} for ${leaderName} ${afterDeck}`.trim());
      }
      return buildPromptWithContext(`${beforeDeck} for ${leaderName}`.trim());
    }
    
    // If prompt starts with a verb like "create", "build", etc., insert leader after the verb
    const verbMatch = userPrompt.match(/^(create|build|design|make|construct)\s+(.*)/i);
    if (verbMatch) {
      const verb = verbMatch[1];
      const rest = verbMatch[2];
      return buildPromptWithContext(`${verb} a deck for ${leaderName} that ${rest}`.trim());
    }
    
    // Default: prepend leader context naturally
    return buildPromptWithContext(`Create a deck for ${leaderName} that ${userPrompt}`.trim());
  }

  // Normalize a card id to canonical format like OP01-001 (drop variant suffixes like _p1, _r1)
  private normalizeCardSetId(id: string): string {
    const upper = (id ?? '').toUpperCase();
    const match = upper.match(/[A-Z]{2}\d{2}-\d{3}/);
    return match ? match[0] : upper;
  }

  /**
   * Suggests leaders using the ML model by generating decks and extracting leader IDs.
   * Generates multiple candidates sequentially to provide variety.
   * Uses greedy decoding for speed (faster than beam search).
   */
  /**
   * Validates if a leader matches the prompt requirements (colors, keywords).
   * Returns a score from 0-1 indicating how well the leader matches.
   * 1.0 = perfect match, 0.0 = no match
   */
  private validateLeaderMatchesPrompt(
    leaderId: string,
    prompt: string,
    requestedColors: Set<string>,
    nonColorKeywords: string[],
  ): { matches: boolean; score: number; reasons: string[] } {
    const metadata = this.leaderKnowledgeService.getLeaderMetadata(leaderId);
    if (!metadata) {
      return { matches: false, score: 0, reasons: ['Leader not found in knowledge base'] };
    }

    const reasons: string[] = [];
    let score = 0;
    let totalChecks = 0;

    // Check color match
    if (requestedColors.size > 0) {
      totalChecks++;
      const leaderColors = new Set(metadata.colors.map((c) => c.toLowerCase()));
      const hasRequestedColor = Array.from(requestedColors).some((c) => leaderColors.has(c));
      if (hasRequestedColor) {
        score += 1.0;
        reasons.push(`✓ Has requested color(s): ${Array.from(requestedColors).join(', ')}`);
      } else {
        reasons.push(`✗ Missing requested color(s): ${Array.from(requestedColors).join(', ')}`);
      }
    }

    // Check keyword match
    if (nonColorKeywords.length > 0) {
      totalChecks++;
      const leaderKeywords = new Set(metadata.keywords.map((k) => k.toLowerCase()));
      const matchedKeywords = nonColorKeywords.filter((k) => leaderKeywords.has(k.toLowerCase()));
      if (matchedKeywords.length > 0) {
        // Partial credit for partial keyword matches
        const keywordScore = matchedKeywords.length / nonColorKeywords.length;
        score += keywordScore;
        reasons.push(`✓ Matches ${matchedKeywords.length}/${nonColorKeywords.length} keywords: ${matchedKeywords.join(', ')}`);
      } else {
        reasons.push(`✗ No keyword matches for: ${nonColorKeywords.join(', ')}`);
      }
    }

    // If no specific requirements, give neutral score
    if (totalChecks === 0) {
      return { matches: true, score: 0.5, reasons: ['No specific requirements to match'] };
    }

    const finalScore = score / totalChecks;
    const matches = finalScore >= 0.5; // Consider it a match if score >= 50%

    return { matches, score: finalScore, reasons };
  }

  async suggestLeaders(
    prompt: string,
    options?: { 
      setIds?: string[]; 
      metaOnly?: boolean; 
      tournamentOnly?: boolean; 
      regenerate?: boolean;
      progressId?: string;
      progressService?: import('./progress.service').ProgressService;
    },
    maxSuggestions: number = 4,
  ): Promise<LeaderSuggestionWithCard[]> {
    if (!this.isEnabled) {
      this.logger.debug('ML leader suggestions skipped: service not enabled or misconfigured');
      return [];
    }

    const progressId = options?.progressId;
    const progress = options?.progressService;

    this.logger.log(`Attempting ML leader suggestions for prompt: "${prompt.substring(0, 50)}..."`);

    const metaOnly = options?.metaOnly ?? false;
    const tournamentOnly = options?.tournamentOnly ?? false;
    
    progress?.updateProgress(progressId!, 'Generating leaders with ML model (trusting 100%)...', 15);

    // TRUST MODEL 100%: No allowlist restrictions - let model generate freely based on prompt
    const leaderIds = new Set<string>();
    
    // Generate leaders freely - let model decide based on prompt
    const maxAttempts = Math.min(maxSuggestions * 3, 8);
    const leaderOnlyTimeout = Math.max(this.config.timeoutMs * 3, 60000);

    // Build leader-only allowlist to restrict first token to leaders only
    // This prevents the model from generating regular cards (like OP02-077, OP12-039, OP04-100)
    // Additionally, use keyword matching to focus on relevant leaders (e.g., "Luffy" -> Luffy leaders)
    // This helps the model when prompts use short names like "Luffy" instead of full "Monkey.D.Luffy"
    const allLeaders = this.leaderKnowledgeService.listLeaderMetadata();
    
    // Try to match prompt keywords to focus the allowlist
    // This is a hybrid approach: we guide the model with relevant leaders, but still trust its choice
    const promptMatch = this.leaderKnowledgeService.matchPrompt(prompt);
    let leaderOnlyAllowlist: string[];
    
    // Separate character/name keywords from color keywords for better matching
    const colorKeywords = new Set(['red', 'blue', 'green', 'purple', 'black', 'yellow']);
    const characterKeywords = new Set<string>();
    const otherKeywords = new Set<string>();
    
    for (const keyword of promptMatch.keywords) {
      if (colorKeywords.has(keyword)) {
        // Color keywords handled separately
      } else {
        // Check if it's a character name (likely if it matches leader names)
        const leaderIdsForKeyword = this.leaderKnowledgeService.getLeaderIdsForKeyword(keyword);
        // Character names typically match 5-30 leaders, not hundreds
        // But be lenient: if it matches < 100 leaders and > 0, consider it a character keyword
        if (leaderIdsForKeyword.length > 0 && leaderIdsForKeyword.length < 100) {
          // Likely a character name (not too broad like "red" which matches 70+ leaders)
          characterKeywords.add(keyword);
        } else {
          otherKeywords.add(keyword);
        }
      }
    }
    
    // Build focused allowlist: prioritize character/name matches, then intersect with colors
    if (characterKeywords.size > 0) {
      // We have character/name keywords - use them as primary filter
      const characterLeaderIds = new Set<string>();
      for (const keyword of characterKeywords) {
        const ids = this.leaderKnowledgeService.getLeaderIdsForKeyword(keyword);
        ids.forEach(id => characterLeaderIds.add(id.toUpperCase()));
      }
      
      // If we also have color keywords, intersect with color matches
      const colorKeywordsInPrompt = Array.from(promptMatch.keywords).filter(k => colorKeywords.has(k));
      if (colorKeywordsInPrompt.length > 0) {
        const colorLeaderIds = new Set<string>();
        for (const color of colorKeywordsInPrompt) {
          const ids = this.leaderKnowledgeService.getLeaderIdsByColor(color);
          ids.forEach(id => colorLeaderIds.add(id.toUpperCase()));
        }
        
        // Intersect: leaders that match BOTH character AND color
        const intersection = new Set<string>();
        for (const id of characterLeaderIds) {
          if (colorLeaderIds.has(id)) {
            intersection.add(id);
          }
        }
        
        if (intersection.size > 0 && intersection.size < allLeaders.length) {
          leaderOnlyAllowlist = Array.from(intersection);
          this.logger.debug(
            `Using intersection allowlist: ${leaderOnlyAllowlist.length} leaders (${Array.from(characterKeywords).join(', ')} AND ${colorKeywordsInPrompt.join(', ')})`,
          );
        } else {
          // Intersection too small or too large - use character matches only
          leaderOnlyAllowlist = Array.from(characterLeaderIds);
          this.logger.debug(
            `Using character-only allowlist: ${leaderOnlyAllowlist.length} leaders (${Array.from(characterKeywords).join(', ')})`,
          );
        }
      } else {
        // Only character keywords, no colors
        leaderOnlyAllowlist = Array.from(characterLeaderIds);
        this.logger.debug(
          `Using character-only allowlist: ${leaderOnlyAllowlist.length} leaders (${Array.from(characterKeywords).join(', ')})`,
        );
      }
    } else if (promptMatch.leaderIds.size > 0 && promptMatch.leaderIds.size < allLeaders.length) {
      // No character keywords, but other keywords matched
      leaderOnlyAllowlist = Array.from(promptMatch.leaderIds).map(id => id.toUpperCase());
      this.logger.debug(
        `Using keyword-matched allowlist: ${leaderOnlyAllowlist.length} leaders (${Array.from(promptMatch.keywords).slice(0, 5).join(', ')})`,
      );
    } else {
      // No keyword matches or too many matches - use all leaders
      leaderOnlyAllowlist = allLeaders.map(l => l.id.toUpperCase());
      this.logger.debug(
        `Using full leader allowlist: ${leaderOnlyAllowlist.length} leaders available (no keyword matches or too broad)`,
      );
    }
    progress?.updateProgress(progressId!, `Generating leaders (0/${maxAttempts} attempts)...`, 20);

    // Generate leaders - model can choose any leader based on prompt, but only leaders allowed
    for (let i = 0; i < maxAttempts && leaderIds.size < maxSuggestions; i++) {
      const request: GenerateDeckRequest = {
        prompt,
        leaderCardId: '', // Empty to let model choose
      };
      
      try {
        const attemptProgress = 20 + Math.floor((i / maxAttempts) * 60);
        progress?.updateProgress(
          progressId!,
          `Generating leader ${i + 1}/${maxAttempts} (${leaderIds.size} found)...`,
          attemptProgress,
        );
        this.logger.debug(`Generation attempt ${i + 1}: Restricting to leaders only (${leaderOnlyAllowlist.length} available)`);
        // Use beam search for better quality
        // Restrict first token to leaders only (prevents generating regular cards)
        const payload = await this.invokePythonModel(request, 'beam', leaderOnlyTimeout, true, leaderOnlyAllowlist);
        const predicted = payload?.leader_id ? this.normalizeCardSetId(String(payload.leader_id)) : '';
        if (predicted) {
          // Verify the generated ID is actually a leader (safety check)
          // This prevents accepting regular cards if the allowlist restriction failed
          const isLeader = allLeaders.some(l => l.id.toUpperCase() === predicted.toUpperCase());
          if (isLeader) {
            leaderIds.add(predicted);
            this.logger.debug(`ML leader generation attempt ${i + 1} succeeded: ${predicted}`);
          } else {
            this.logger.warn(
              `ML leader generation attempt ${i + 1} generated non-leader card: ${predicted}. This should not happen with allowlist restriction.`,
            );
          }
        }
      } catch (error) {
        this.logger.debug(`ML leader generation attempt ${i + 1} failed: ${(error as Error).message}`);
      }
    }

    if (leaderIds.size === 0) {
      this.logger.warn('ML leader suggestions: no valid leaders generated.');
      return [];
    }

    // Fetch leader cards (canonical ids only)
    progress?.updateProgress(progressId!, 'Finalizing leader suggestions...', 85);
    const uniqueLeaderIds = Array.from(leaderIds).slice(0, maxSuggestions);
    const leaderCards = await this.optcgApiService.getCardsBySetIds(uniqueLeaderIds);
    const cardMap = new Map(leaderCards.map((card) => [card.id.toUpperCase(), card]));

    const results: LeaderSuggestionWithCard[] = [];
    for (const leaderId of uniqueLeaderIds) {
      const card = cardMap.get(leaderId.toUpperCase());
      // TRUST MODEL 100%: Accept all generated leaders without filtering
      if (!card) {
        continue;
      }

      results.push({
        cardSetId: card.id.toUpperCase(),
        leaderName: card.name,
        rationale: `Generated by ML model based on prompt: "${prompt}"`,
        card,
      });
    }

    this.logger.log(`✓ Leader suggestions generated via ML model (trusting 100%, count: ${results.length})`);
    return results;
  }

  private async leaderMeetsFilters(leaderId: string, requestedSetIds: string[], metaOnly: boolean, tournamentOnly: boolean = false): Promise<boolean> {
    if (this.cardKnowledgeService.isBannedCard(leaderId)) {
      return false;
    }
    if (metaOnly && !this.cardKnowledgeService.isMetaCard(leaderId)) {
      return false;
    }
    if (tournamentOnly && !(await this.cardKnowledgeService.isTournamentCard(leaderId))) {
      return false;
    }
    if (!requestedSetIds.length) {
      return true;
    }
    const setCode = this.getSetCode(leaderId);
    const normalizedSetCode = this.normalizeSetCode(setCode);
    const normalizedRequested = requestedSetIds.map((id) => this.normalizeSetCode(id));
    const matches = normalizedRequested.includes(normalizedSetCode);
    if (!matches && leaderId.startsWith('OP13')) {
      // Debug logging for OP13 leaders to diagnose the issue
      this.logger.debug(
        `Leader ${leaderId} set code mismatch: ${normalizedSetCode} (from ${setCode}) vs requested: ${normalizedRequested.join(', ')}`,
      );
    }
    return matches;
  }

  private normalizeSetCode(setCode: string): string {
    return setCode.toUpperCase().replace(/-/g, '');
  }

  private getSetCode(cardId: string): string {
    const canonical = this.normalizeCardSetId(cardId);
    const [setCode] = canonical.split('-');
    return setCode ?? canonical;
  }
}
