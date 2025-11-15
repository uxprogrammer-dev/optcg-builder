import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import OpenAI from 'openai';
import {
  DeckCardSuggestion,
  GenerateDeckRequest,
  LeaderSuggestion,
} from '@shared/contracts/ai';
import { CardKnowledgeService } from './card-knowledge.service';
import { RulesLoaderService } from './rules-loader.service';
import { LeaderStrategyService } from './leader-strategy.service';

interface LeaderSuggestionResponse {
  leaders: LeaderSuggestion[];
}

interface DeckSuggestionResponse {
  summary: string;
  cards: DeckCardSuggestion[];
}

@Injectable()
export class OpenAiService {
  private readonly client: OpenAI;
  private readonly logger = new Logger(OpenAiService.name);

  private readonly model: string;
  private readonly temperature: number;

  constructor(
    private readonly configService: ConfigService,
    private readonly cardKnowledgeService: CardKnowledgeService,
    private readonly rulesLoaderService: RulesLoaderService,
    private readonly leaderStrategyService: LeaderStrategyService,
  ) {
    const apiKey = this.configService.get<string>('openai.apiKey');
    if (!apiKey) {
      throw new Error('OPENAI_API_KEY is not configured');
    }

    this.model = this.configService.get<string>('openai.model') ?? 'gpt-4o-mini';
    this.temperature = this.configService.get<number>('openai.temperature') ?? 0.3;

    this.client = new OpenAI({ apiKey });
  }

  // Normalize to canonical set id like OP01-001 (drop any suffixes / free text)
  private normalizeCardSetId(id: string): string | null {
    const upper = (id ?? '').toUpperCase();
    const match = upper.match(/[A-Z]{2}\d{2}-\d{3}/);
    return match ? match[0] : null;
  }

  private getSetCode(cardId: string): string {
    const canonical = this.normalizeCardSetId(cardId) ?? cardId.toUpperCase();
    const [setCode] = canonical.split('-');
    return setCode ?? canonical;
  }

  private normalizeSetCode(setCode: string): string {
    return setCode.toUpperCase().replace(/-/g, '');
  }

  private mergeAndValidateCards(cards: DeckCardSuggestion[]): DeckCardSuggestion[] {
    const merged = new Map<string, DeckCardSuggestion>();
    for (const card of cards) {
      const norm = this.normalizeCardSetId(card.cardSetId);
      if (!norm) {
        continue; // drop invalid IDs like 'TBD' or names
      }
      const prev = merged.get(norm);
      const quantity = Math.min(4, (prev?.quantity ?? 0) + Math.max(1, Math.min(4, card.quantity ?? 1)));
      merged.set(norm, {
        cardSetId: norm,
        quantity,
        role: card.role,
        rationale: card.rationale ?? undefined,
      });
    }

    const result = Array.from(merged.values());
    result.sort((a, b) => (b.quantity ?? 0) - (a.quantity ?? 0) || a.cardSetId.localeCompare(b.cardSetId));
    return result;
  }

  async suggestLeaders(prompt: string): Promise<LeaderSuggestionResponse> {
    const schema = {
      type: 'object',
      additionalProperties: false,
      required: ['leaders'],
      properties: {
        leaders: {
          type: 'array',
          minItems: 1,
          maxItems: 4,
          items: {
            type: 'object',
            additionalProperties: false,
            required: ['cardSetId', 'leaderName', 'rationale', 'strategySummary'],
            properties: {
              cardSetId: { type: 'string', pattern: '^[A-Z0-9-]+$' },
              leaderName: {
                anyOf: [{ type: 'string', minLength: 1 }, { type: 'null' }],
              },
              rationale: {
                anyOf: [{ type: 'string', minLength: 1 }, { type: 'null' }],
              },
              strategySummary: {
                anyOf: [{ type: 'string', minLength: 1 }, { type: 'null' }],
              },
            },
          },
        },
      },
    };

    const systemPrompt = [
      'You are an expert One Piece Trading Card Game deck builder.',
      'Analyze the player prompt and recommend existing leader cards.',
      'Use official card set identifiers such as OP01-001.',
      'Provide clear strategic rationales referencing mechanics and archetypes.',
      'Only suggest leaders that actually exist in the card database.',
    ].join(' ');

    return this.generateJson<LeaderSuggestionResponse>({
      schemaName: 'LeaderSuggestions',
      schema,
      systemPrompt,
      userPrompt: prompt,
    });
  }

  async generateDeck({
    prompt,
    leaderCardId,
    setIds,
    metaOnly,
    tournamentOnly,
  }: GenerateDeckRequest): Promise<DeckSuggestionResponse> {
    const schema = {
      type: 'object',
      additionalProperties: false,
      required: ['summary', 'cards'],
      properties: {
        summary: { type: 'string' },
        cards: {
          type: 'array',
          minItems: 10,
          maxItems: 100,
          items: {
            type: 'object',
            additionalProperties: false,
            required: ['cardSetId', 'quantity', 'role', 'rationale'],
            properties: {
              cardSetId: { type: 'string', pattern: '^[A-Z0-9-]+$' },
              quantity: { type: 'integer', minimum: 1, maximum: 4 },
              role: {
                type: 'string',
                enum: ['character', 'event', 'stage', 'don', 'counter', 'other'],
              },
              rationale: {
                anyOf: [{ type: 'string', minLength: 1 }, { type: 'null' }],
              },
            },
          },
        },
      },
    };

    const rulesPrimer = await this.rulesLoaderService.getRulesForPrompt({
      includeSummary: true,
      maxSources: 1,
      maxSectionsPerSource: 4,
      maxParagraphsPerSection: 2,
    });

    // Get leader-specific strategy if available
    const leaderStrategy = await this.leaderStrategyService.getStrategyDescription(leaderCardId);
    const strategyContext = leaderStrategy
      ? `\nLeader Strategy: ${leaderStrategy}\n`
      : '';

    const systemPrompt = [
      'You are an expert One Piece TCG deck constructor.',
      '',
      rulesPrimer,
      strategyContext,
      'Given the selected leader and player prompt, craft a 50 card decklist (not counting DON).',
      'Only include cards that synergize with the leader ability, share critical subtypes, or reflect proven tournament pairings.',
      'Reject any card whose color identity, conditional text, or leader requirements conflict with the chosen leader.',
      'Each inclusion must have a clear role (engine, removal, protection, finisher, combo extender) and support other cards in the list.',
      'Good selections: cards that search for leader-specific pieces, enable the leader ability, or combo with existing archetype cores.',
      'Bad selections: off-color tech cards, vanilla fillers with no strategic purpose, or cards that require incompatible leader types.',
      'Ensure card set identifiers are accurate and available in current sets.',
      'Respect deck building rules: max 4 copies of each card, match leader color identity, and follow color legality.',
      'Include DON cards only if explicitly relevant, otherwise omit them.',
      'Provide a concise strategy summary.',
    ].join('\n');

    const deckPromptLines = [
      `Selected leader: ${leaderCardId}`,
      `User request: ${prompt}`,
    ];
    if (metaOnly) {
      deckPromptLines.push('Use only competitive meta cards commonly seen in top tournament decks.');
    }
    if (setIds?.length) {
      deckPromptLines.push(`Allowed sets: ${setIds.join(', ')} (avoid cards from other sets).`);
    }

    const deckPrompt = deckPromptLines.join('\n');

    const raw = await this.generateJson<DeckSuggestionResponse>({
      schemaName: 'DeckSuggestion',
      schema,
      systemPrompt,
      userPrompt: deckPrompt,
    });

    // Post-process: normalize IDs and merge duplicates, drop invalids
    const cleanedCards = this.mergeAndValidateCards(raw.cards ?? []);
    const allowedSetIds = new Set((setIds ?? []).map((id) => this.normalizeSetCode(id)));
    const tournamentOnlyFlag = tournamentOnly ?? false;
    
    // Filter synchronously first
    let filteredCards = cleanedCards.filter((card) => {
      if (this.cardKnowledgeService.isBannedCard(card.cardSetId)) {
        return false;
      }
      if (metaOnly && !this.cardKnowledgeService.isMetaCard(card.cardSetId)) {
        return false;
      }
      if (allowedSetIds.size > 0) {
        const setCode = this.getSetCode(card.cardSetId);
        const normalizedSetCode = this.normalizeSetCode(setCode);
        if (!allowedSetIds.has(normalizedSetCode)) {
          return false;
        }
      }
      return true;
    });

    // Apply tournament filter if enabled (async)
    if (tournamentOnlyFlag) {
      const tournamentChecks = await Promise.all(
        filteredCards.map((card) => this.cardKnowledgeService.isTournamentCard(card.cardSetId)),
      );
      filteredCards = filteredCards.filter((_, index) => tournamentChecks[index]);
    }
    return {
      summary: raw.summary,
      cards: filteredCards,
    };
  }

  private async generateJson<T>({
    systemPrompt,
    userPrompt,
    schemaName,
    schema,
  }: {
    systemPrompt: string;
    userPrompt: string;
    schemaName: string;
    schema: Record<string, unknown>;
  }): Promise<T> {
    try {
      const requestPayload = {
        model: this.model,
        input: `${systemPrompt}\n\nUser prompt:\n${userPrompt}`,
        text: {
          format: {
            type: 'json_schema',
            name: schemaName,
            schema,
            strict: true,
          },
        },
      } as const;

      if (this.shouldSendTemperature()) {
        (requestPayload as any).temperature = this.temperature;
      }

      const response = await this.client.responses.create(requestPayload as any);

      const json =
        (response as { output_text?: string }).output_text ??
        (response as any)?.output?.[0]?.content?.[0]?.text ??
        '';

      if (!json?.trim()) {
        throw new Error('Empty response from OpenAI');
      }

      return JSON.parse(json) as T;
    } catch (error) {
      this.logger.error(`OpenAI request failed: ${(error as Error).message}`, error as Error);
      throw error;
    }
  }

  private shouldSendTemperature(): boolean {
    if (this.temperature === undefined || this.temperature === null) {
      return false;
    }

    const model = this.model.toLowerCase();
    if (model.startsWith('gpt-5')) {
      return false;
    }

    return true;
  }

  async analyzeDeck(
    prompt: string,
    leader: import('@shared/types/optcg-card').OptcgCard,
    cards: import('@shared/contracts/ai').DeckCardSuggestionWithCard[],
  ): Promise<string> {
    const cardList = cards
      .map((c) => `${c.cardSetId} x${c.quantity} - ${c.card.name}${c.card.text ? ` (${c.card.text.substring(0, 100)}...)` : ''}`)
      .join('\n');

    const totalCards = cards.reduce((sum, c) => sum + c.quantity, 0);
    const characterCount = cards
      .filter((c) => c.role === 'character')
      .reduce((sum, c) => sum + c.quantity, 0);
    const eventCount = cards
      .filter((c) => c.role === 'event')
      .reduce((sum, c) => sum + c.quantity, 0);
    const stageCount = cards
      .filter((c) => c.role === 'stage')
      .reduce((sum, c) => sum + c.quantity, 0);

    const rulesPrimer = await this.rulesLoaderService.getRulesForPrompt({
      includeSummary: true,
      maxSources: 2,
      maxSectionsPerSource: 5,
      maxParagraphsPerSection: 2,
    });

    const systemPrompt = [
      'You are an expert One Piece Trading Card Game deck analyst.',
      '',
      rulesPrimer,
      'Analyze the provided deck list and provide a comprehensive review covering:',
      '- Highlight how each key card supports the leader ability, archetype foundations, or common tournament synergies.',
      '- Call out any card whose colors or conditional requirements conflict with the leader, and suggest sharper replacements.',
      '1. Overall deck strategy and synergy',
      '2. Card ratios and deck composition (Characters, Events, Stages)',
      '3. Strengths and potential weaknesses',
      '4. Key combos and interactions',
      '5. Gameplay Guide: Detailed turn-by-turn gameplay instructions, including:',
      '   - Early game (Turns 1-2): What to play, what to search for, key setup plays',
      '   - Mid game (Turns 3-4): How to develop your board, when to use key cards',
      '   - Late game (Turns 5+): How to close out games, finisher plays, win conditions',
      '6. Strategies: Advanced tactics and decision-making tips, including:',
      '   - Resource management (Don!! cards, hand size, life management)',
      '   - When to be aggressive vs defensive',
      '   - Key decision points and what to prioritize',
      '   - How to adapt to different matchups',
      '7. Tips and Tricks: Practical advice for playing this deck, including:',
      '   - Common mistakes to avoid',
      '   - Optimal sequencing of cards',
      '   - How to maximize leader ability usage',
      '   - Sideboard considerations (if applicable)',
      '8. Suggestions for improvement (if any)',
      '9. Competitive viability assessment',
      '',
      'Format your response with clear sections:',
      '- Use numbered headings for main sections (e.g., "1. Overall Strategy")',
      '- Use bullet points (- or •) for lists of items',
      '- Use short headings ending with colons (:) for subsections',
      '- Separate paragraphs with blank lines',
      '- Keep paragraphs concise and focused',
      '- Make the Gameplay Guide, Strategies, and Tips sections detailed and actionable',
      '',
      'Provide a detailed, professional analysis that would help a player understand, play, and optimize this deck.',
    ].join('\n');

    const userPrompt = [
      `Original Prompt: ${prompt}`,
      '',
      `Leader: ${leader.name} (${leader.id})`,
      leader.text ? `Leader Ability: ${leader.text}` : 'Leader has no special ability',
      '',
      `Deck List (${totalCards} cards total):`,
      `- Characters: ${characterCount} cards`,
      `- Events: ${eventCount} cards`,
      `- Stages: ${stageCount} cards`,
      '',
      'Cards:',
      cardList,
      '',
      'Please provide a comprehensive deck analysis.',
    ].join('\n');

    try {
      const startTime = Date.now();
      const response = await this.client.chat.completions.create({
        model: this.model,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userPrompt },
        ],
        temperature: 0.7, // Slightly higher temperature for more creative analysis
        max_tokens: 3000, // Increased to accommodate gameplay, strategies, and tips sections
      });

      const duration = Date.now() - startTime;
      this.logger.log(`OpenAI deck analysis completed in ${duration}ms`);

      const review = response.choices[0]?.message?.content?.trim() ?? '';
      if (!review) {
        throw new Error('Empty review from OpenAI');
      }

      return review;
    } catch (error) {
      this.logger.error(`OpenAI deck analysis failed: ${(error as Error).message}`, error as Error);
      throw error;
    }
  }
}

