import { Injectable, Logger } from '@nestjs/common';
import {
  DeckCardSuggestion,
  GenerateDeckRequest,
  LeaderSuggestionWithCard,
} from '@shared/contracts/ai';
import { OptcgApiService } from '../optcg/optcg-api.service';
import { OpenAiService } from './openai.service';
import {
  LeaderIntentMatch,
  PromptIntentAnalysis,
} from './intent-matcher.service';
import { MlIntentService } from './ml-intent.service';
import { LocalDeckBuilderService } from './local-deck-builder.service';
import { MlDeckService } from './ml-deck.service';
import { LeaderKnowledgeService } from './leader-knowledge.service';
import { OptcgCard } from '@shared/types/optcg-card';
import { CardKnowledgeService } from './card-knowledge.service';
import { ProgressService } from './progress.service';

export type DeckSuggestionResult = {
  summary: string;
  cards: DeckCardSuggestion[];
  source: 'local' | 'ml' | 'openai';
  notes?: string[];
  usedKeywords?: string[];
  gameplaySummary?: string;
  comboHighlights?: string[];
};

@Injectable()
export class AiService {
  private readonly logger = new Logger(AiService.name);
  private readonly localLeaderScoreThreshold = 10;

  constructor(
    private readonly openAiService: OpenAiService,
    private readonly optcgApiService: OptcgApiService,
    private readonly intentMatcher: MlIntentService,
    private readonly localDeckBuilder: LocalDeckBuilderService,
    private readonly leaderKnowledge: LeaderKnowledgeService,
    private readonly mlDeckService: MlDeckService,
    private readonly cardKnowledge: CardKnowledgeService,
    private readonly progressService?: ProgressService,
  ) {}

  async suggestLeaders(
    prompt: string,
    useOpenAi: boolean = false,
    options?: { 
      setIds?: string[]; 
      metaOnly?: boolean; 
      tournamentOnly?: boolean; 
      regenerate?: boolean;
      progressId?: string;
      progressService?: ProgressService;
    },
  ): Promise<LeaderSuggestionWithCard[]> {
    const progressId = options?.progressId;
    const progress = options?.progressService;
    
    this.logger.log(
      `suggestLeaders called (useOpenAi=${useOpenAi}) - Always using ML first; metaOnly=${options?.metaOnly}, sets=${options?.setIds?.length ?? 0}, regenerate=${options?.regenerate}`,
    );
    
    progress?.updateProgress(progressId!, 'Analyzing prompt...', 5);
    const analysis = await this.intentMatcher.analyzePrompt(prompt);

    // Always prioritize ML for leader suggestions, regardless of useOpenAi flag
    this.logger.debug('Attempting ML model for leader suggestions...');
    progress?.updateProgress(progressId!, 'Loading ML model...', 10);
    const mlSuggestions = await this.mlDeckService.suggestLeaders(prompt, {
      setIds: options?.setIds,
      metaOnly: options?.metaOnly ?? false,
      tournamentOnly: options?.tournamentOnly ?? false,
      regenerate: options?.regenerate ?? false,
      progressId,
      progressService: progress,
    });
    // TRUST MODEL 100%: Return ML suggestions without any filtering
    if (mlSuggestions.length) {
      this.logger.log(
        `✓ Leader suggestions generated via ML transformer model (trusting 100%, count: ${mlSuggestions.length})`,
      );
      return mlSuggestions;
    }

    // Only use OpenAI if explicitly requested by user
    if (useOpenAi) {
      this.logger.debug('ML returned no suggestions. Using OpenAI as requested...');
      try {
        const openAiLeaders = await this.buildOpenAiLeaderSuggestions(prompt, analysis, {
          setIds: undefined,
          metaOnly: options?.metaOnly ?? false,
          tournamentOnly: options?.tournamentOnly ?? false,
        });
        if (openAiLeaders.length) {
          this.logger.log(`✓ Leader suggestions generated via OpenAI (count: ${openAiLeaders.length})`);
          return openAiLeaders;
        }
      } catch (error) {
        this.logger.warn(`OpenAI leader suggestions failed: ${(error as Error).message}`);
      }
    }

    this.logger.warn('No leader suggestions available from ML model.');
    return [];
  }

  async generateDeckSuggestion(
    request: GenerateDeckRequest,
    useOpenAi: boolean = false,
    progressId?: string,
    progressService?: ProgressService,
  ): Promise<DeckSuggestionResult> {
    // Only use OpenAI if explicitly requested by user
    if (useOpenAi) {
      this.logger.debug('Attempting OpenAI deck generation (requested)...');
      progressService?.updateProgress(progressId!, 'Generating deck with OpenAI...', 20);
      try {
        const openAiDeck = await this.openAiService.generateDeck(request);
        this.logger.log(`✓ Deck generated via OpenAI (source: openai)`);
        return {
          summary: openAiDeck.summary,
          cards: openAiDeck.cards,
          source: 'openai',
        };
      } catch (error) {
        this.logger.warn(`OpenAI deck generation failed: ${(error as Error).message}`);
        // Don't fall back to ML if OpenAI was explicitly requested
        return {
          summary: 'OpenAI deck generation failed. Please try again.',
          cards: [],
          source: 'openai',
        };
      }
    }

    // Always use ML model (trust 100%)
    this.logger.debug('Attempting ML deck generation...');
    progressService?.updateProgress(progressId!, 'Loading ML model...', 20);
    const mlDeck = await this.mlDeckService.generateDeck(request, progressId, progressService);
    if (mlDeck) {
      this.logger.log(`✓ Deck generated via ML transformer model (trusting 100%, source: ml)`);
      return mlDeck;
    }

    this.logger.warn('ML deck generation failed.');
    return {
      summary: 'Unable to generate a deck for this prompt. Please try again with a different description.',
      cards: [],
      source: 'ml',
      gameplaySummary: `Unable to derive a game plan from: ${request.prompt.trim()}`,
    };
  }

  private async buildLocalLeaderSuggestions(
    analysis: PromptIntentAnalysis,
  ): Promise<LeaderSuggestionWithCard[]> {
    if (!analysis.leaderMatches.length) {
      return [];
    }

    const topMatch = analysis.leaderMatches[0];
    const hasExplicitRequest = analysis.explicitLeaderIds.length > 0;
    const hasColorSignal = topMatch.matchedColors.length > 0 || analysis.colors.length > 0;
    const hasKeywordSignal = topMatch.matchedKeywords.length > 0;

    if (!hasExplicitRequest) {
      const meetsBaseThreshold = topMatch.score >= this.localLeaderScoreThreshold;
      const meetsColorThreshold = hasColorSignal && topMatch.score >= 4;
      const meetsKeywordThreshold = hasKeywordSignal && topMatch.score >= 6;

      if (!meetsBaseThreshold && !meetsColorThreshold && !meetsKeywordThreshold) {
        return [];
      }
    }

    const matches = analysis.leaderMatches.slice(0, 4);
    const cards = await this.optcgApiService.getCardsBySetIds(
      matches.map((match) => match.leaderId),
    );
    const cardMap = new Map(cards.map((card) => [card.id.toUpperCase(), card]));

    const suggestions: LeaderSuggestionWithCard[] = [];
    for (const match of matches) {
      const card = cardMap.get(match.leaderId.toUpperCase());
      if (!card || card.type.toLowerCase() !== 'leader') {
        continue;
      }

      suggestions.push({
        cardSetId: card.id,
        leaderName: card.name,
        rationale: this.composeLocalLeaderRationale(match),
        strategySummary: this.composeStrategySummary(match, analysis),
        card,
      });
    }

    return this.dedupeLeaders(suggestions);
  }

  private composeLocalLeaderRationale(match: LeaderIntentMatch): string {
    const parts: string[] = [];
    if (match.matchedColors.length) {
      parts.push(`Matches requested colors (${match.matchedColors.join(', ')})`);
    }
    if (match.matchedKeywords.length) {
      parts.push(`Aligns with prompt themes: ${match.matchedKeywords.join(', ')}`);
    }
    if (match.metadata.subtypes.length) {
      parts.push(`Supports ${match.metadata.subtypes.join(', ')}`);
    }
    parts.push(`Confidence score ${match.score.toFixed(1)}`);
    return parts.join('. ');
  }

  private composeStrategySummary(match: LeaderIntentMatch, analysis: PromptIntentAnalysis): string {
    const keywordFocus = match.matchedKeywords.slice(0, 3);
    if (keywordFocus.length) {
      return `${match.leaderName} plan focuses on ${keywordFocus.join(', ')} synergies.`;
    }

    const colors = match.matchedColors.length ? match.matchedColors : analysis.colors;
    if (colors.length) {
      const label = colors.map((color) => color.charAt(0).toUpperCase() + color.slice(1)).join('/');
      return `${match.leaderName} leverages ${label} color identity.`;
    }

    if (match.metadata.subtypes.length) {
      return `${match.leaderName} builds around ${match.metadata.subtypes.join(', ')}.`;
    }

    return `${match.leaderName} offers the strongest alignment with the described deck plan.`;
  }

  private async buildOpenAiLeaderSuggestions(
    prompt: string,
    analysis: PromptIntentAnalysis,
    options?: { setIds?: string[]; metaOnly?: boolean; tournamentOnly?: boolean },
  ): Promise<LeaderSuggestionWithCard[]> {
    const { leaders } = await this.openAiService.suggestLeaders(prompt);
    if (!leaders.length) {
      return [];
    }

    const cards = await Promise.all(
      leaders.map((leader) => this.fetchLeaderCardWithFallback(leader.cardSetId)),
    );
    const cardMap = new Map(
      cards.filter((card): card is OptcgCard => Boolean(card)).map((card) => [card.id.toLowerCase(), card]),
    );

    const enriched = leaders.reduce<LeaderSuggestionWithCard[]>((acc, leader) => {
      const card = cardMap.get(leader.cardSetId.toLowerCase());
      if (!card || card.type?.toLowerCase() !== 'leader') {
        return acc;
      }

      acc.push({
        cardSetId: leader.cardSetId,
        leaderName: leader.leaderName ?? card.name,
        rationale: leader.rationale ?? '',
        strategySummary: leader.strategySummary ?? undefined,
        card,
      });

      return acc;
    }, []);

    const dedupedOpenAi = this.dedupeLeaders(enriched);
    const filtered = this.filterAndSortOpenAiLeaders(dedupedOpenAi, analysis);

    let results = this.dedupeLeaders(filtered.length ? filtered : dedupedOpenAi);
    // Do NOT filter by sets for leader suggestions - check all matching leaders across all sets
    // Only apply meta-only and tournament-only filters if requested
    if (options) {
      const filterResults = await Promise.all(
        results.map((leader) =>
          this.leaderMeetsFilters(leader.card, undefined, options.metaOnly ?? false, options.tournamentOnly ?? false),
        ),
      );
      results = results.filter((_, index) => filterResults[index]);
    }

    // Ensure explicitly requested leaders appear first if possible
    const preferredIds = new Set(
      [...analysis.explicitLeaderIds, ...analysis.cardIds].map((id) => this.canonicalizeSetId(id)),
    );

    const presentIds = new Set(results.map((leader) => this.canonicalizeSetId(leader.card.id)));
    const missingPreferred = Array.from(preferredIds).filter((id) => !presentIds.has(id));

    if (missingPreferred.length) {
      const inserted: LeaderSuggestionWithCard[] = [];
      for (const id of missingPreferred) {
        try {
          const card = await this.fetchLeaderCardWithFallback(id);
          if (!card || card.type?.toLowerCase() !== 'leader') {
            continue;
          }
          inserted.push({
            cardSetId: card.id,
            leaderName: card.name,
            rationale: `Explicitly referenced in prompt: ${prompt.slice(0, 80)}${prompt.length > 80 ? '…' : ''}`,
            strategySummary: `${card.name} matches the requested leader.`,
            card,
          });
        } catch (error) {
          this.logger.debug(`Unable to load preferred leader ${id}: ${(error as Error).message}`);
        }
      }
      if (inserted.length) {
        results = this.dedupeLeaders([...inserted, ...results]);
      }
    }

    if (!results.length) {
      return [];
    }

    return results.slice(0, 4);
  }

  private filterAndSortOpenAiLeaders(
    leaders: LeaderSuggestionWithCard[],
    analysis: PromptIntentAnalysis,
  ): LeaderSuggestionWithCard[] {
    const desiredIds = new Set<string>([...analysis.explicitLeaderIds, ...analysis.cardIds]);
    const keywordSet = new Set<string>(analysis.keywords);
    const colorSet = new Set<string>(analysis.colors.map((color) => color.toLowerCase()));
    const scoreById = new Map<string, number>(
      analysis.leaderMatches.map((match) => [match.leaderId.toUpperCase(), match.score]),
    );

    const filtered = leaders.filter((leader) => {
      const leaderId = leader.card.id.toUpperCase();
      if (desiredIds.size && !desiredIds.has(leaderId)) {
        return false;
      }

      if (colorSet.size) {
        const cardColors = (leader.card.color ?? '')
          .toLowerCase()
          .split(/[\/,\s]+/)
          .filter(Boolean);
        const matchesColors = Array.from(colorSet).every((color) => cardColors.includes(color));
        if (!matchesColors) {
          return false;
        }
      }

      if (keywordSet.size) {
        const matchesKeyword = Array.from(keywordSet).some((keyword) =>
          this.leaderKnowledge.keywordMatchesLeader(keyword, leader.card.id),
        );
        if (!matchesKeyword && !desiredIds.size) {
          return false;
        }
      }

      return true;
    });

    const ordered = (filtered.length ? filtered : leaders).slice();
    ordered.sort((a, b) => {
      const scoreA = scoreById.get(a.card.id.toUpperCase()) ?? 0;
      const scoreB = scoreById.get(b.card.id.toUpperCase()) ?? 0;
      if (scoreA === scoreB) {
        return 0;
      }
      return scoreB - scoreA;
    });

    return ordered;
  }

  private dedupeLeaders(leaders: LeaderSuggestionWithCard[]): LeaderSuggestionWithCard[] {
    const deduped: LeaderSuggestionWithCard[] = [];
    const seen = new Set<string>();

    for (const leader of leaders) {
      const canonicalId = this.canonicalizeSetId(leader.card.id);
      if (seen.has(canonicalId)) {
        continue;
      }
      seen.add(canonicalId);
      deduped.push({
        ...leader,
        cardSetId: canonicalId,
      });
    }

    return deduped;
  }

  private canonicalizeSetId(cardId: string): string {
    const match = cardId.toUpperCase().match(/^([A-Z0-9]+-[0-9]+)/);
    return match ? match[1] : cardId.toUpperCase();
  }

  private getSetCode(cardId: string): string {
    const canonical = this.canonicalizeSetId(cardId);
    const [setCode] = canonical.split('-');
    return setCode ?? canonical;
  }

  private async ensureExplicitLeaders(
    leaders: LeaderSuggestionWithCard[],
    analysis: PromptIntentAnalysis,
    prompt: string,
    setIds?: string[],
    metaOnly?: boolean,
    tournamentOnly?: boolean,
  ): Promise<LeaderSuggestionWithCard[]> {
    const preferredIds = new Set<string>(
      [...analysis.explicitLeaderIds, ...analysis.cardIds].map((id) => this.canonicalizeSetId(id)),
    );
    if (!preferredIds.size) {
      return leaders;
    }

    const seen = new Set(leaders.map((leader) => this.canonicalizeSetId(leader.card.id)));
    const insertions: LeaderSuggestionWithCard[] = [];

    for (const id of preferredIds) {
      if (seen.has(id)) {
        continue;
      }
      const card = await this.fetchLeaderCardWithFallback(id);
      if (!card || card.type?.toLowerCase() !== 'leader') {
        continue;
      }
      // Do NOT filter by sets for leader suggestions - only check banned/meta/tournament
      if (!(await this.leaderMeetsFilters(card, undefined, metaOnly ?? false, tournamentOnly ?? false))) {
        continue;
      }
      insertions.push({
        cardSetId: card.id,
        leaderName: card.name,
        rationale: `Explicitly referenced in prompt: "${prompt.slice(0, 80)}${prompt.length > 80 ? '…' : ''}"`,
        strategySummary: `${card.name} matches the requested leader.`,
        card,
      });
      seen.add(id);
    }

    if (!insertions.length) {
      return leaders;
    }

    return this.dedupeLeaders([...insertions, ...leaders]);
  }

  private limitToExplicitLeaders(
    leaders: LeaderSuggestionWithCard[],
    analysis: PromptIntentAnalysis,
  ): LeaderSuggestionWithCard[] {
    const explicitSet = new Set(
      [...analysis.explicitLeaderIds, ...analysis.cardIds].map((id) => this.canonicalizeSetId(id)),
    );
    if (!explicitSet.size) {
      return leaders;
    }
    const filtered = leaders.filter((leader) => explicitSet.has(this.canonicalizeSetId(leader.card.id)));
    return filtered.length ? filtered : leaders;
  }

  private async fetchLeaderCardWithFallback(cardSetId: string): Promise<OptcgCard | null> {
    const canonical = this.canonicalizeSetId(cardSetId);
    const fromApi = await this.optcgApiService.getCardBySetId(canonical).catch(() => null);
    if (fromApi) {
      return fromApi;
    }
    const metadata = this.leaderKnowledge.getLeaderMetadata(canonical);
    if (!metadata) {
      return null;
    }
    return {
      id: canonical,
      name: metadata.name,
      text: null,
      type: 'Leader',
      color: metadata.colors.join('/'),
      cost: null,
      power: null,
      life: null,
      attribute: null,
      subtypes: metadata.subtypes,
      rarity: null,
      setId: canonical.split('-')[0] ?? '',
      setName: metadata.setName ?? '',
      imageUrl: '',
      marketPrice: null,
      inventoryPrice: null,
      raw: null as any,
    };
  }

  private async leaderMeetsFilters(
    leader: OptcgCard,
    requestedSetIds: string[] | undefined,
    metaOnly: boolean,
    tournamentOnly: boolean = false,
  ): Promise<boolean> {
    if (this.cardKnowledge.isBannedCard(leader.id)) {
      return false;
    }

    if (metaOnly && !this.cardKnowledge.isMetaCard(leader.id)) {
      return false;
    }

    if (tournamentOnly && !(await this.cardKnowledge.isTournamentCard(leader.id))) {
      return false;
    }

    if (!requestedSetIds || !requestedSetIds.length) {
      return true;
    }

    const leaderSetId = this.getSetCode(leader.id);
    const normalizedLeaderSetId = this.normalizeSetCode(leaderSetId);
    return requestedSetIds.some((id) => this.normalizeSetCode(id) === normalizedLeaderSetId);
  }

  private normalizeSetCode(setCode: string): string {
    return setCode.toUpperCase().replace(/-/g, '');
  }
}
