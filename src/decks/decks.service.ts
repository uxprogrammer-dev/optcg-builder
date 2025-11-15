import { BadRequestException, Injectable, NotFoundException, Logger } from '@nestjs/common';
import {
  DeckCardSuggestionWithCard,
  GenerateDeckRequest,
  GeneratedDeck,
  CardSuggestionResponse,
  DeckCharacteristicId,
  ReviewDeckResponse,
} from '@shared/contracts/ai';
import { AiService } from '../ai/ai.service';
import { OptcgApiService } from '../optcg/optcg-api.service';
import { DeckRegistryService } from './deck-registry.service';
import { CardKnowledgeService } from '../ai/card-knowledge.service';
import { OpenAiService } from '../ai/openai.service';
import { TournamentSynergyService } from '../ai/tournament-synergy.service';
import { CardTierService } from '../ai/card-tier.service';
import { LeaderStrategyService } from '../ai/leader-strategy.service';
import { SuggestCardDto } from './dto/suggest-card.dto';
import { ReviewDeckDto } from './dto/review-deck.dto';

const CHARACTERISTIC_KEYWORDS: Record<DeckCharacteristicId, string[]> = {
  control: ['control', 'board', 'active', 'rest', 'tap', 'set up', 'stun', 'lock'],
  cardAdvantage: ['draw', 'search', 'add to hand', 'look at', 'reveal', 'tutor'],
  removal: ['destroy', 'k.o.', 'remove', 'banish', 'bounce', 'retire', 'return to hand', 'bottom of deck'],
  trashManipulation: ['trash', 'graveyard', 'discard pile', 'from trash', 'from your trash', 'bottom of your deck'],
  aggression: ['attack', 'damage', 'double attack', 'rush', 'power up'],
  defense: ['blocker', 'counter', 'life', 'prevent', 'reduce damage', 'guard', 'shield', 'heal', 'cannot be k.o.'],
  synergy: ['when', 'if', 'combo', 'trigger', 'activate', 'support', 'ally', 'partner'],
  economy: ['cost', 'reduce cost', 'don', 'don!!', 'gain don', 'set don', 'resource'],
};

const LEADER_ABILITY_SYNERGY_RULES: Array<{
  abilityPattern: RegExp;
  supportKeywords: string[];
}> = [
  { abilityPattern: /(draw|add to (?:your )?hand|search)/i, supportKeywords: ['draw', 'search', 'add', 'hand', 'tutor', 'reveal'] },
  { abilityPattern: /(don|don!!|set don|rested don)/i, supportKeywords: ['don', 'rest', 'set don', 'gain don', 'reduce cost'] },
  { abilityPattern: /(rest|active|restand|stand)/i, supportKeywords: ['rest', 'active', 'restand', 'stand', 'unrest', 'tap', 'untap'] },
  { abilityPattern: /(k\.?o\.?|destroy|remove)/i, supportKeywords: ['k.o.', 'destroy', 'remove', 'banish', 'bounce'] },
  { abilityPattern: /(counter|blocker|defend|guard)/i, supportKeywords: ['counter', 'blocker', 'guard', 'shield', 'prevent', 'reduce'] },
  { abilityPattern: /(life|recover)/i, supportKeywords: ['life', 'recover', 'heal', 'gain life', 'trigger'] },
  { abilityPattern: /(trash|graveyard|discard)/i, supportKeywords: ['trash', 'discard', 'graveyard', 'from your trash', 'return from trash'] },
  { abilityPattern: /(combo|trigger|activate)/i, supportKeywords: ['combo', 'trigger', 'activate', 'support'] },
];

@Injectable()
export class DecksService {
  private readonly logger = new Logger(DecksService.name);

  constructor(
    private readonly aiService: AiService,
    private readonly optcgApiService: OptcgApiService,
    private readonly deckRegistry: DeckRegistryService,
    private readonly cardKnowledge: CardKnowledgeService,
    private readonly openAiService: OpenAiService,
    private readonly tournamentSynergy: TournamentSynergyService,
    private readonly cardTierService: CardTierService,
    private readonly leaderStrategyService: LeaderStrategyService,
  ) {}

  async generateDeck({
    prompt,
    leaderCardId,
    useOpenAi,
    setIds,
    metaOnly,
    tournamentOnly,
    progressId,
    progressService,
  }: GenerateDeckRequest & { 
    useOpenAi?: boolean; 
    setIds?: string[]; 
    metaOnly?: boolean; 
    tournamentOnly?: boolean;
    progressId?: string;
    progressService?: import('../ai/progress.service').ProgressService;
  }): Promise<GeneratedDeck & {
    totalCards: number;
  }> {
    progressService?.updateProgress(progressId!, 'Loading leader card...', 5);
    const leader = await this.optcgApiService.getCardBySetId(leaderCardId);

    const allowedSetIds = new Set((setIds ?? []).map((id) => this.normalizeSetCode(id)));
    const metaOnlyFlag = metaOnly ?? false;
    const tournamentOnlyFlag = tournamentOnly ?? false;

    progressService?.updateProgress(progressId!, 'Generating deck with ML model...', 15);
    const suggestion = await this.aiService.generateDeckSuggestion(
      { prompt, leaderCardId, setIds: Array.from(allowedSetIds), metaOnly: metaOnlyFlag, tournamentOnly: tournamentOnlyFlag },
      useOpenAi ?? false,
      progressId,
      progressService,
    );
    const suggestedCardIds = suggestion.cards.map((card) => card.cardSetId);
    
    this.logger.debug(
      `ML/AI suggested ${suggestedCardIds.length} cards. Fetching from API...`,
    );
    progressService?.updateProgress(progressId!, `Fetching ${suggestedCardIds.length} cards from API...`, 70);

    const cards = await this.optcgApiService.getCardsBySetIds(suggestedCardIds);
    this.logger.debug(
      `Fetched ${cards.length} cards from API (requested ${suggestedCardIds.length}). Missing: ${suggestedCardIds.length - cards.length}`,
    );
    const cardMap = new Map(cards.map((card) => [card.id.toLowerCase(), card]));

    const leaderColors = new Set(this.parseColors(leader?.color));
    const leaderSubtypes = new Set(
      (leader?.subtypes ?? []).map((s) => s.toLowerCase().trim()).filter(Boolean),
    );

    let cardsNotFound = 0;
    
    const enrichedCards = suggestion.cards.reduce<DeckCardSuggestionWithCard[]>(
      (acc, cardSuggestion) => {
        const card = cardMap.get(cardSuggestion.cardSetId.toLowerCase());
        if (!card) {
          cardsNotFound++;
          return acc;
        }

        // TRUST MODEL 100%: Accept all cards without any filtering
        // Keep model's original quantities - it learned from tournament data
        const role = this.resolveRole(card);
        acc.push({
          ...cardSuggestion,
          quantity: Math.min(cardSuggestion.quantity ?? 1, 4),
          rationale: cardSuggestion.rationale ?? '',
          role,
          card,
        });

        return acc;
      },
      [],
    );
    
    this.logger.debug(
      `Enrichment complete: ${enrichedCards.length} cards enriched. ` +
      `Not found in API: ${cardsNotFound}. ` +
      `TRUSTING MODEL 100%: No filtering applied.`,
    );

    // Merge card variants before normalization
    let workingCards = this.mergeCardVariants(enrichedCards);
    
    this.logger.debug(
      `TRUSTING MODEL 100%: ${workingCards.length} cards from model (no filtering). Source: ${suggestion.source}`,
    );
    
    // TRUST MODEL 100%: Never add fallback cards - use exactly what the model generated
    const isMLSource = suggestion.source === 'ml';
    
    // Only add fallback if we have 0 cards (model completely failed)
    if (workingCards.length === 0) {
      if (isMLSource) {
        this.logger.debug(`ML deck has very few cards (${workingCards.length}), adding minimal fallback cards only if necessary...`);
      } else {
        this.logger.debug(`Too few cards (${workingCards.length}), building fallback cards with progressive relaxation...`);
      }
      // First attempt: strict filters (set, meta, color, archetype, no vanilla)
      workingCards = await this.buildColorFallbackCards(
        leaderColors,
        leaderCardId,
        workingCards, // Include any cards we already have
        leaderSubtypes,
        leader,
        prompt,
        false, // no vanilla
        { allowedSetIds, metaOnly: metaOnlyFlag },
      );
      workingCards = this.mergeCardVariants(workingCards);
      workingCards = workingCards.filter((entry) => this.cardMeetsFilters(entry.card, allowedSetIds, metaOnlyFlag, leader));
      this.logger.debug(`After first fallback attempt: ${workingCards.length} cards`);
      
      // If still too few, relax archetype alignment and conditional effects
      if (workingCards.length < 5) {
        this.logger.debug(`Still too few cards (${workingCards.length}), relaxing archetype alignment and conditional effects...`);
        // Try with a more lenient buildColorFallbackCards that skips conditional effects check
        const relaxedCards = await this.buildColorFallbackCardsLenient(
          leaderColors,
          leaderCardId,
          workingCards,
          undefined, // No archetype restriction
          leader, // Pass leader for rationale generation
          false, // no vanilla yet
          { allowedSetIds, metaOnly: metaOnlyFlag },
        );
        const relaxedMerged = this.mergeCardVariants(relaxedCards);
        const relaxedFiltered = relaxedMerged.filter((entry) => this.cardMeetsFilters(entry.card, allowedSetIds, metaOnlyFlag, leader));
        // Merge with existing cards, avoiding duplicates
        const existingIds = new Set(workingCards.map(c => this.getBaseCardCode(c.card.id)));
        for (const card of relaxedFiltered) {
          if (!existingIds.has(this.getBaseCardCode(card.card.id))) {
            workingCards.push(card);
            existingIds.add(this.getBaseCardCode(card.card.id));
          }
        }
        this.logger.debug(`After relaxing archetype alignment and conditional effects: ${workingCards.length} cards`);
      }
      
      // Don't add vanilla cards - prefer cards with abilities that synergize
      // If still too few, try one more time with more lenient filters but still no vanilla
      if (workingCards.length < 5) {
        this.logger.debug(`Still too few cards (${workingCards.length}), trying with more lenient filters (no vanilla)...`);
        const additionalCards = await this.buildColorFallbackCardsLenient(
          leaderColors,
          leaderCardId,
          workingCards,
          undefined, // No archetype restriction
          leader, // Pass leader for rationale generation
          false, // Still no vanilla - prefer cards with abilities
          { allowedSetIds, metaOnly: metaOnlyFlag },
        );
        const additionalMerged = this.mergeCardVariants(additionalCards);
        const additionalFiltered = additionalMerged
          .filter((entry) => this.cardMeetsFilters(entry.card, allowedSetIds, metaOnlyFlag, leader))
          .filter((entry) => !this.isVanillaCard(entry.card)); // Explicitly filter vanilla
        // Merge with existing cards, avoiding duplicates
        const existingIds = new Set(workingCards.map(c => this.getBaseCardCode(c.card.id)));
        for (const card of additionalFiltered) {
          if (!existingIds.has(this.getBaseCardCode(card.card.id))) {
            workingCards.push(card);
            existingIds.add(this.getBaseCardCode(card.card.id));
          }
        }
        this.logger.debug(`After additional non-vanilla cards: ${workingCards.length} cards`);
      }
    }
    
    this.logger.debug(`Final working cards before normalization: ${workingCards.length} cards`);

    if (!workingCards.length) {
      throw new BadRequestException(
        'Unable to generate a deck for the selected leader. Please try a different prompt or leader.',
      );
    }

    const normalizedCards = this.normalizeDeckCards(workingCards);

    // For ML-generated decks, trust the model's learned quantities from tournament data
    // Only fill to 50 if we're significantly below (e.g., < 40 cards)
    // For non-ML sources, fill to 50 as before
    let currentTotal = normalizedCards.reduce((sum, c) => sum + c.quantity, 0);
    const fillThreshold = isMLSource ? 40 : 50; // ML: only fill if < 40, others: fill to 50
    
    if (currentTotal < fillThreshold) {
      if (isMLSource) {
        this.logger.debug(`ML deck has ${currentTotal} cards (threshold: ${fillThreshold}), trusting model's learned quantities. Only filling if significantly below threshold.`);
        // For ML decks, be more conservative with quantity increases - trust the model's learned quantities
        // Only increase if we're significantly below the threshold (e.g., < 35)
        if (currentTotal < 35) {
          this.logger.debug(`ML deck has ${currentTotal} cards, increasing quantities conservatively to reach minimum threshold...`);
          currentTotal = await this.increaseExistingCardQuantities(
            normalizedCards,
            currentTotal,
            leaderCardId,
            leader,
            prompt,
            false, // Normal mode - conservative increases
          );
        } else {
          this.logger.debug(`ML deck has ${currentTotal} cards (>= 35), trusting model's learned quantities without aggressive increases.`);
        }
      } else {
        // Non-ML: Aggressively increase quantities to improve consistency
        currentTotal = await this.increaseExistingCardQuantities(
          normalizedCards,
          currentTotal,
          leaderCardId,
          leader,
          prompt,
          false, // First pass: normal mode
        );
      }

      // Last resort: Add new filler cards ONLY if all existing cards are already at 4 copies
      // For ML decks, be even more conservative - only add fillers if we're very far below threshold
      const fillerThreshold = isMLSource ? 35 : 50; // ML: only if < 35, others: if < 50
      if (currentTotal < fillerThreshold) {
        // Check if all existing cards are already at maximum (4 copies)
        const allCardsMaxed = normalizedCards.every((card) => card.quantity >= 4);
        
        if (allCardsMaxed) {
          // All cards are at 4, so we need to add new filler cards
          // For ML decks, only add fillers if we're very far below threshold
          if (isMLSource && currentTotal >= 35) {
            this.logger.debug(`ML deck has ${currentTotal} cards with all cards maxed, but above minimum threshold (35). Trusting model's deck size.`);
          } else {
            await this.addFillerCardsMatchingLeaderType(
              normalizedCards,
              currentTotal,
              leaderColors,
              leaderCardId,
              leaderSubtypes,
              leader,
              prompt,
              { allowedSetIds, metaOnly: metaOnlyFlag },
            );
            // Recalculate total after adding fillers
            currentTotal = normalizedCards.reduce((sum, c) => sum + c.quantity, 0);
          }
        } else {
          // Try one more time to increase existing cards
          // For ML decks, be conservative - only if we're well below threshold
          if (isMLSource && currentTotal >= 35) {
            this.logger.debug(`ML deck has ${currentTotal} cards, above minimum threshold. Not aggressively increasing quantities.`);
          } else {
            currentTotal = await this.increaseExistingCardQuantities(
              normalizedCards,
              currentTotal,
              leaderCardId,
              leader,
              prompt,
              !isMLSource, // ML: normal mode, others: aggressive mode
            );
          }
          
          // If we still couldn't reach threshold and all cards are now maxed, add fillers
          const finalFillerThreshold = isMLSource ? 35 : 50;
          if (currentTotal < finalFillerThreshold) {
            const allCardsMaxedNow = normalizedCards.every((card) => card.quantity >= 4);
            if (allCardsMaxedNow) {
              // For ML decks, only add fillers if we're very far below threshold
              if (isMLSource && currentTotal >= 35) {
                this.logger.debug(`ML deck has ${currentTotal} cards with all cards maxed, but above minimum threshold (35). Trusting model's deck size.`);
              } else {
                await this.addFillerCardsMatchingLeaderType(
                  normalizedCards,
                  currentTotal,
                  leaderColors,
                  leaderCardId,
                  leaderSubtypes,
                  leader,
                  prompt,
                  { allowedSetIds, metaOnly: metaOnlyFlag },
                );
                currentTotal = normalizedCards.reduce((sum, c) => sum + c.quantity, 0);
              }
            }
          }
        }
      }
    } else if (isMLSource && currentTotal < 50) {
      // ML deck is between 40-50 cards - this is acceptable, trust the model
      this.logger.debug(`ML deck has ${currentTotal} cards (between 40-50), trusting model's learned deck size from tournament data.`);
    }

    // Final normalization to guarantee constraints (<=50 total, <=4 each)
    const finalCards = this.normalizeDeckCards(normalizedCards);
    
    // For ML-generated decks, use minimal post-processing to preserve model's learned choices
    // The model was trained on 7700+ tournament decks and knows optimal card selections
    let ratioAdjustedCards: DeckCardSuggestionWithCard[];
    let finalTotal: number;
    
    if (suggestion.source === 'ml') {
      // TRUST MODEL 100%: No filtering, no post-processing - use exactly what the model generated
      this.logger.debug('TRUSTING MODEL 100%: No post-processing or filtering applied to ML-generated deck');
      
      // Use all cards exactly as the model generated them
      let mlCards = finalCards;
      
      // Enhance card rationales with tier and strategy information (non-destructive, doesn't change cards)
      mlCards = await this.enhanceCardRationales(mlCards, leaderCardId, leader);
      
      // Only normalize quantities (cap at 4) - don't change model's learned quantities
      ratioAdjustedCards = mlCards.map((card) => ({
        ...card,
        quantity: Math.min(card.quantity, 4),
      }));
      
      finalTotal = ratioAdjustedCards.reduce((sum, c) => sum + c.quantity, 0);
      
      // TRUST MODEL 100%: Never adjust quantities or add cards - use exactly what model generated
      this.logger.debug(`TRUSTING MODEL 100%: Using ${finalTotal} cards exactly as model generated (no adjustments)`);
    } else {
      // Full post-processing for non-ML sources (OpenAI, local builder)
      const themedCards = this.enforceThemeDistribution(prompt, leader, finalCards);
      
      // Filter out cards that don't contribute to deck purpose
      const relevantCards = await this.filterRelevantCards(themedCards, prompt, leader);
      
      // Remove vanilla cards completely - they don't contribute to competitive decks
      // Each card must have a purpose or synergy with the deck/leader
      const finalCardsBeforeConsolidation = relevantCards.filter((c) => !this.isVanillaCard(c.card));
      
      // Consolidate x1 cards - reduce singleton cards and boost important ones
      const consolidatedCards = this.consolidateSingletonCards(finalCardsBeforeConsolidation, prompt, leader);
      
      // Enhance card rationales with tier and strategy information
      const enhancedCards = await this.enhanceCardRationales(consolidatedCards, leaderCardId, leader);
      
      // Enforce proper deck ratios (Characters 60-70%, Events 15-25%, Stages 2-4% if needed)
      ratioAdjustedCards = await this.enforceDeckRatios(
        enhancedCards,
        leaderColors,
        leaderCardId,
        leaderSubtypes,
        leader,
        prompt,
        { allowedSetIds, metaOnly: metaOnlyFlag },
      );
      ratioAdjustedCards = ratioAdjustedCards.filter((card) =>
        this.cardMeetsFilters(card.card, allowedSetIds, metaOnlyFlag, leader),
      );
      finalTotal = ratioAdjustedCards.reduce((sum, c) => sum + c.quantity, 0);

      // Ensure balanced color distribution for multi-color leaders
      if (leaderColors.size > 1) {
        ratioAdjustedCards = await this.ensureColorBalance(
          ratioAdjustedCards,
          leaderColors,
          leaderCardId,
          leaderSubtypes,
          leader,
          prompt,
          { allowedSetIds, metaOnly: metaOnlyFlag },
        );
        finalTotal = ratioAdjustedCards.reduce((sum, c) => sum + c.quantity, 0);
      }
    }

    // For non-ML sources, apply aggressive filler logic if needed
    // For ML sources, we already handled this above (only if < 40 cards)
    if (suggestion.source !== 'ml' && finalTotal < 50) {
      // First: Intelligently increase quantities of existing cards
      finalTotal = await this.increaseExistingCardQuantities(
        ratioAdjustedCards,
        finalTotal,
        leaderCardId,
        leader,
        prompt,
        false, // First pass: normal mode
      );

      // Last resort: Add new filler cards ONLY if all existing cards are already at 4 copies
      if (finalTotal < 50) {
        // Check if all existing cards are already at maximum (4 copies)
        const allCardsMaxed = ratioAdjustedCards.every((card) => card.quantity >= 4);
        
        if (allCardsMaxed) {
          // All cards are at 4, so we need to add new filler cards
          await this.addFillerCardsMatchingLeaderType(
            ratioAdjustedCards,
            finalTotal,
            leaderColors,
            leaderCardId,
            leaderSubtypes,
            leader,
            prompt,
            { allowedSetIds, metaOnly: metaOnlyFlag },
          );
          // Recalculate total after adding fillers
          finalTotal = ratioAdjustedCards.reduce((sum, c) => sum + c.quantity, 0);
        } else {
          // Try one more time to increase existing cards (more aggressively)
          finalTotal = await this.increaseExistingCardQuantities(
            ratioAdjustedCards,
            finalTotal,
            leaderCardId,
            leader,
            prompt,
            true, // Second pass: aggressive mode - prioritize all cards including 1-ofs
          );
          
          // If we still couldn't reach 50 and all cards are now maxed, add fillers
          if (finalTotal < 50) {
            const allCardsMaxedNow = ratioAdjustedCards.every((card) => card.quantity >= 4);
            if (allCardsMaxedNow) {
              await this.addFillerCardsMatchingLeaderType(
                ratioAdjustedCards,
                finalTotal,
                leaderColors,
                leaderCardId,
                leaderSubtypes,
                leader,
                prompt,
                { allowedSetIds, metaOnly: metaOnlyFlag },
              );
              finalTotal = ratioAdjustedCards.reduce((sum, c) => sum + c.quantity, 0);
            }
          }
        }
      }
    }

    const gameplaySummary =
      suggestion.gameplaySummary ??
      this.buildGameplaySummary(prompt, leader, ratioAdjustedCards);
    const comboHighlights =
      suggestion.comboHighlights?.length
        ? suggestion.comboHighlights
        : this.buildComboHighlights(prompt, leader, ratioAdjustedCards);

    // Always generate a deck review with gameplay, strategies, and tips
    let deckReview: string | undefined;
    if (useOpenAi) {
      try {
        this.logger.log('Generating OpenAI deck review...');
        deckReview = await this.openAiService.analyzeDeck(prompt, leader, ratioAdjustedCards);
        this.logger.log('✓ OpenAI deck review generated');
      } catch (error) {
        this.logger.warn(`OpenAI deck review failed: ${(error as Error).message}. Falling back to basic review.`);
        // Fall back to basic review if OpenAI fails
        deckReview = this.buildBasicDeckReview(prompt, leader, ratioAdjustedCards, gameplaySummary, comboHighlights);
      }
    } else {
      // Generate basic review even when OpenAI is not enabled
      deckReview = this.buildBasicDeckReview(prompt, leader, ratioAdjustedCards, gameplaySummary, comboHighlights);
    }

    const deckRecord = this.deckRegistry.createDeck({
      prompt,
      summary: suggestion.summary,
      leaderCardId,
      leader,
      cards: ratioAdjustedCards,
      gameplaySummary,
      comboHighlights,
      source: suggestion.source,
      notes: suggestion.notes,
      deckReview,
      filters: {
        setIds: Array.from(allowedSetIds),
        metaOnly: metaOnlyFlag,
      },
    });

    return {
      deckId: deckRecord.deckId,
      summary: deckRecord.summary,
      leader: deckRecord.leader,
      cards: deckRecord.cards,
      totalCards: finalTotal,
      source: suggestion.source,
      notes: suggestion.notes,
      gameplaySummary: deckRecord.gameplaySummary,
      comboHighlights: deckRecord.comboHighlights,
      deckReview: deckRecord.deckReview,
    };
  }

  async suggestCards(dto: SuggestCardDto): Promise<CardSuggestionResponse> {
    const {
      leaderCardId,
      cardIdQuery,
      characteristicId,
      excludeCardIds = [],
      limit = 10,
      setIds,
      metaOnly,
    } = dto;

    const allowedSetIds = new Set((setIds ?? []).map((id) => this.normalizeSetCode(id)));
    const metaOnlyFlag = metaOnly ?? false;

    const leader = await this.optcgApiService.getCardBySetId(leaderCardId);
    if (!leader) {
      throw new BadRequestException(`Leader ${leaderCardId} not found`);
    }

    const leaderColors = new Set(this.parseColors(leader.color));
    const excludeBaseCodes = new Set(excludeCardIds.map((id) => this.getBaseCardCode(id)));
    const normalizedCardIdQuery = cardIdQuery?.trim().toUpperCase();

    const metadataList = await this.cardKnowledge.listAllMetadata();

    const candidates: {
      metadataId: string;
      rawScore: number;
      normalizedScore: number;
      matchedKeywords: string[];
    }[] = [];

    metadataList.forEach((metadata) => {
      if (metadata.type?.toLowerCase() === 'leader') {
        return;
      }

      if (!this.metadataMeetsFilters(metadata.id, allowedSetIds, metaOnlyFlag)) {
        return;
      }

      const baseCode = this.getBaseCardCode(metadata.id);
      if (excludeBaseCodes.has(baseCode)) {
        return;
      }

      if (!this.metadataIsColorLegal(metadata.colors, leaderColors)) {
        return;
      }

      const matchesIdQuery = normalizedCardIdQuery
        ? metadata.id.toUpperCase().includes(normalizedCardIdQuery) || metadata.name.toUpperCase().includes(normalizedCardIdQuery)
        : false;

      if (normalizedCardIdQuery && !matchesIdQuery) {
        return;
      }

      if (characteristicId) {
        const { score, matchedKeywords } = this.computeCharacteristicScoreFromMetadata(
          metadata.text,
          metadata.type ?? '',
          characteristicId,
        );
        if (score <= 0) {
          return;
        }
        const normalized = Number(((1 - Math.exp(-score / 4)) * 10).toFixed(2));
        candidates.push({
          metadataId: metadata.id,
          rawScore: score,
          normalizedScore: normalized,
          matchedKeywords,
        });
      } else if (matchesIdQuery) {
        candidates.push({
          metadataId: metadata.id,
          rawScore: 100,
          normalizedScore: 10,
          matchedKeywords: ['ID match'],
        });
      }
    });

    if (!candidates.length) {
      return {
        characteristicId,
        cards: [],
      };
    }

    const sorted = candidates
      .sort((a, b) => b.rawScore - a.rawScore || a.metadataId.localeCompare(b.metadataId))
      .slice(0, Math.min(Math.max(limit, 1), 30));

    const ids = sorted.map((candidate) => candidate.metadataId);
    const cards = await this.optcgApiService.getCardsBySetIds(ids);
    const cardMap = new Map(cards.map((card) => [card.id.toUpperCase(), card]));

    const suggestions = sorted
      .map((candidate) => {
        const card = cardMap.get(candidate.metadataId.toUpperCase());
        if (!card) {
          return null;
        }
        if (!this.cardMeetsFilters(card, allowedSetIds, metaOnlyFlag, leader)) {
          return null;
        }
        return {
          card,
          normalizedScore: candidate.normalizedScore,
          rawScore: candidate.rawScore,
          characteristicId,
          rationale: characteristicId
            ? `Highlights: ${candidate.matchedKeywords.join(', ')}`
            : `Matches ID query ${normalizedCardIdQuery}`,
        };
      })
      .filter((value): value is NonNullable<typeof value> => value !== null);

    return {
      characteristicId,
      cards: suggestions,
    };
  }

  async reviewDeck(dto: ReviewDeckDto): Promise<ReviewDeckResponse> {
    const { prompt, leaderCardId, cards } = dto;

    if (!cards || cards.length === 0) {
      throw new BadRequestException('No cards provided for deck review.');
    }

    const totalCards = cards.reduce((sum, card) => sum + card.quantity, 0);
    if (totalCards !== 50) {
      throw new BadRequestException(`Deck review requires exactly 50 cards. Provided: ${totalCards}.`);
    }

    const leader = await this.optcgApiService.getCardBySetId(leaderCardId);
    if (!leader) {
      throw new BadRequestException(`Leader ${leaderCardId} not found.`);
    }

    const cardIds = cards.map((card) => card.cardSetId);
    const fetchedCards = await this.optcgApiService.getCardsBySetIds(cardIds);
    const cardMap = new Map(fetchedCards.map((card) => [card.id.toUpperCase(), card]));

    const deckCards: DeckCardSuggestionWithCard[] = [];
    cards.forEach((entry) => {
      const card = cardMap.get(entry.cardSetId.toUpperCase());
      if (!card) {
        this.logger.warn(`Card ${entry.cardSetId} not found when re-running deck review.`);
        return;
      }
      deckCards.push({
        cardSetId: card.id,
        quantity: entry.quantity,
        role: this.resolveRole(card),
        rationale: 'User-adjusted card',
        card,
      });
    });

    if (deckCards.length === 0) {
      throw new BadRequestException('No valid cards found for deck review.');
    }

    const review = await this.openAiService.analyzeDeck(prompt, leader, deckCards);
    return { deckReview: review };
  }

  private metadataIsColorLegal(cardColors: string[], leaderColors: Set<string>): boolean {
    if (!cardColors.length) {
      return true;
    }
    return cardColors.every((color) => leaderColors.has(color.toLowerCase()));
  }

  private computeCharacteristicScoreFromMetadata(
    text: string | null,
    type: string,
    characteristicId: DeckCharacteristicId,
  ): { score: number; matchedKeywords: string[] } {
    const keywords = CHARACTERISTIC_KEYWORDS[characteristicId] ?? [];
    const haystack = `${text ?? ''}`.toLowerCase();
    const matchedKeywords = new Set<string>();
    let score = 0;

    keywords.forEach((keyword) => {
      const hits = this.countKeywordHits(haystack, keyword);
      if (hits > 0) {
        matchedKeywords.add(keyword);
        score += hits;
      }
    });

    const normalizedType = type.toLowerCase();
    if (characteristicId === 'aggression' && normalizedType === 'character' && haystack.includes('rush')) {
      matchedKeywords.add('rush');
      score += 1;
    }
    if (characteristicId === 'defense' && normalizedType === 'event' && haystack.includes('counter')) {
      matchedKeywords.add('counter');
      score += 1;
    }
    if (characteristicId === 'economy' && haystack.includes('don')) {
      matchedKeywords.add('don');
      score += 1;
    }

    return {
      score,
      matchedKeywords: Array.from(matchedKeywords).slice(0, 5),
    };
  }

  private countKeywordHits(text: string, keyword: string): number {
    if (!keyword) {
      return 0;
    }
    const escaped = keyword.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const regex = new RegExp(escaped, 'g');
    return (text.match(regex) ?? []).length;
  }

  getDeck(deckId: string): GeneratedDeck & { totalCards: number } {
    const deck = this.deckRegistry.getDeck(deckId);
    if (!deck) {
      throw new NotFoundException(`Deck ${deckId} not found`);
    }

    const totalCards = deck.cards.reduce((total, current) => total + current.quantity, 0);

    return {
      deckId: deck.deckId,
      summary: deck.summary,
      leader: deck.leader,
      cards: deck.cards,
      totalCards,
      source: deck.source,
      notes: deck.notes,
      gameplaySummary: deck.gameplaySummary,
      comboHighlights: deck.comboHighlights,
      deckReview: deck.deckReview,
    };
  }

  /**
   * Gets the base card code by removing variant suffixes (_p1, _p2, _r1, etc.)
   * Example: "OP02-085_p1" -> "OP02-085"
   */
  private getBaseCardCode(cardId: string): string {
    // Remove variant suffixes like _p1, _p2, _r1, etc.
    // Match pattern: base code (e.g., OP02-085) followed by optional variant suffix
    const match = cardId.match(/^([A-Z0-9]+-[0-9]+)/i);
    return match ? match[1].toUpperCase() : cardId.toUpperCase();
  }

  private getSetCode(cardId: string): string {
    const match = cardId.toUpperCase().match(/^([A-Z0-9]+)-/);
    return match ? match[1] : '';
  }

  private normalizeSetCode(setCode: string): string {
    return setCode.toUpperCase().replace(/-/g, '');
  }

  /**
   * Parses leader ability text for cost restrictions.
   * Returns max allowed cost if restriction found, null otherwise.
   */
  private parseLeaderCostRestriction(leader: import('@shared/types/optcg-card').OptcgCard): { maxCost?: number; minCost?: number } | null {
    const abilityText = (leader.text ?? '').toLowerCase();
    
    // Pattern: "cannot include cards with a cost of X or more"
    // Pattern: "cannot include cards with cost X or more"
    const maxCostPattern = /cannot\s+include\s+cards?\s+with\s+(?:a\s+)?cost\s+of\s+(\d+)\s+or\s+more/i;
    const maxCostMatch = abilityText.match(maxCostPattern);
    if (maxCostMatch) {
      const maxCost = parseInt(maxCostMatch[1], 10);
      return { maxCost: maxCost - 1 }; // If "5 or more" is banned, max allowed is 4
    }
    
    // Pattern: "cannot include cards with cost X or less" (less common)
    const minCostPattern = /cannot\s+include\s+cards?\s+with\s+(?:a\s+)?cost\s+of\s+(\d+)\s+or\s+less/i;
    const minCostMatch = abilityText.match(minCostPattern);
    if (minCostMatch) {
      const minCost = parseInt(minCostMatch[1], 10);
      return { minCost: minCost + 1 }; // If "2 or less" is banned, min allowed is 3
    }
    
    return null;
  }

  /**
   * Checks if a card meets leader-specific restrictions (cost limits, etc.)
   */
  private cardMeetsLeaderRestrictions(
    card: import('@shared/types/optcg-card').OptcgCard,
    leader: import('@shared/types/optcg-card').OptcgCard,
  ): boolean {
    const costRestriction = this.parseLeaderCostRestriction(leader);
    if (!costRestriction) {
      return true; // No restrictions
    }
    
    const cardCost = parseInt(card.cost ?? '0', 10);
    
    if (costRestriction.maxCost !== undefined && cardCost > costRestriction.maxCost) {
      return false; // Card cost exceeds maximum allowed
    }
    if (costRestriction.minCost !== undefined && cardCost < costRestriction.minCost) {
      return false; // Card cost below minimum allowed
    }
    
    return true;
  }

  private cardMeetsFilters(
    card: import('@shared/types/optcg-card').OptcgCard, 
    allowedSetIds: Set<string>, 
    metaOnly: boolean,
    leader?: import('@shared/types/optcg-card').OptcgCard
  ): boolean {
    if (this.cardKnowledge.isBannedCard(card.id)) {
      return false;
    }
    if (metaOnly && !this.cardKnowledge.isMetaCard(card.id)) {
      return false;
    }
    if (allowedSetIds.size > 0) {
      const setCode = this.getSetCode(card.id);
      const normalizedSetCode = this.normalizeSetCode(setCode);
      if (!allowedSetIds.has(normalizedSetCode)) {
        return false;
      }
    }
    
    // Check leader-specific restrictions (cost limits, etc.)
    if (leader && !this.cardMeetsLeaderRestrictions(card, leader)) {
      return false;
    }
    
    return true;
  }

  /**
   * Merges cards with the same base code (variants) into a single entry
   */
  private mergeCardVariants(cards: DeckCardSuggestionWithCard[]): DeckCardSuggestionWithCard[] {
    const merged = new Map<string, DeckCardSuggestionWithCard>();

    for (const card of cards) {
      const baseCode = this.getBaseCardCode(card.cardSetId);
      const existing = merged.get(baseCode);

      if (existing) {
        // Merge quantities, but cap at 4 total
        const newQuantity = Math.min(existing.quantity + card.quantity, 4);
        existing.quantity = newQuantity;
        // Keep the first variant's card data (or prefer the one with higher quantity)
        if (card.quantity > existing.quantity) {
          existing.card = card.card;
          existing.cardSetId = card.cardSetId;
        }
      } else {
        // First occurrence of this base code
        merged.set(baseCode, { ...card, quantity: Math.min(card.quantity, 4) });
      }
    }

    return Array.from(merged.values());
  }

  private normalizeDeckCards(cards: DeckCardSuggestionWithCard[]): DeckCardSuggestionWithCard[] {
    // First, merge card variants (same base code)
    const merged = this.mergeCardVariants(cards);
    const cloned = merged.map((card) => ({ ...card }));
    let total = cloned.reduce((sum, card) => sum + card.quantity, 0);

    if (total === 50) {
      return cloned;
    }

    if (total < 50) {
      let remaining = 50 - total;
      // Sort by importance: prioritize cards with higher current quantity and important roles
      const sorted = cloned.slice().sort((a, b) => {
        // Prefer cards that already have more copies (they're more important)
        if (b.quantity !== a.quantity) return b.quantity - a.quantity;
        // Then prefer characters and events
        const rolePriority: Record<string, number> = { character: 3, event: 2, stage: 1, don: 0, counter: 0, other: 0 };
        const aPriority = rolePriority[a.role ?? 'other'] ?? 0;
        const bPriority = rolePriority[b.role ?? 'other'] ?? 0;
        return bPriority - aPriority;
      });
      
      // Distribute remaining cards, prioritizing important cards
      for (const card of sorted) {
        if (remaining <= 0) break;
        const available = Math.min(4 - card.quantity, remaining);
        if (available > 0) {
          card.quantity += available;
          remaining -= available;
        }
      }
      // If still short due to insufficient unique cards, return best-effort deck (no error)
      return cloned;
    } else {
      let excess = total - 50;
      const sorted = cloned
        .slice()
        .sort((a, b) => b.quantity - a.quantity || a.card.name.localeCompare(b.card.name));

      for (const card of sorted) {
        if (excess <= 0) break;
        const reducible = Math.min(card.quantity - 1, excess);
        if (reducible > 0) {
          card.quantity -= reducible;
          excess -= reducible;
        }
      }

      // If still excess, keep trimming by reducing to minimum 1 in order
      for (const card of sorted) {
        if (excess <= 0) break;
        if (card.quantity > 1) {
          const reducible = Math.min(card.quantity - 1, excess);
          card.quantity -= reducible;
          excess -= reducible;
        }
      }

      return sorted.sort((a, b) => a.cardSetId.localeCompare(b.cardSetId));
    }
  }

  private parseColors(value?: string | null): string[] {
    return (value ?? '')
      .toLowerCase()
      .split(/[\s/&,+]+/)
      .map((c) => c.trim())
      .filter((c) => !!c);
  }

  private isColorLegal(cardColor: string | null | undefined, leaderColors: Set<string>): boolean {
    const colors = this.parseColors(cardColor);
    if (colors.length === 0) {
      return true;
    }
    // A card is legal if ALL of its colors are in the leader's colors
    // For example, if leader is "Red/Green":
    // - "Red" card is legal (Red is in leaderColors)
    // - "Green" card is legal (Green is in leaderColors)
    // - "Red/Green" card is legal (both are in leaderColors)
    // - "Blue" card is NOT legal (Blue is not in leaderColors)
    // - "Red/Blue" card is NOT legal (Blue is not in leaderColors)
    return colors.every((c) => leaderColors.has(c));
  }

  /**
   * Checks if a card is vanilla (has no ability/text).
   * Vanilla cards are cards without any text or ability.
   */
  private isVanillaCard(card: import('@shared/types/optcg-card').OptcgCard): boolean {
    const text = (card.text ?? '').trim();
    // Card is vanilla if it has no text/ability
    return text.length === 0;
  }

  private isArchetypeAligned(
    card: import('@shared/types/optcg-card').OptcgCard,
    allowedSubtypes: Set<string>,
  ): boolean {
    // If no subtypes required, allow the card (for backwards compatibility)
    if (allowedSubtypes.size === 0) {
      return true;
    }

    const cardSubtypes = (card.subtypes ?? []).map((s) => s.toLowerCase().trim()).filter(Boolean);
    if (cardSubtypes.length === 0) {
      // If card has no subtypes, only allow if no subtypes are required
      return false;
    }

    // Card must share at least one subtype with allowed subtypes
    return cardSubtypes.some((subtype) => allowedSubtypes.has(subtype));
  }

  /**
   * Checks if a card's conditional effects can be used with the selected leader.
   * Filters out cards that require leader types/subtypes that don't match.
   */
  private canUseCardEffects(
    card: import('@shared/types/optcg-card').OptcgCard,
    leader: import('@shared/types/optcg-card').OptcgCard,
  ): boolean {
    const cardText = (card.text ?? '').toLowerCase();
    
    // If card has no conditional text, it can always be used
    if (!cardText || cardText.length === 0) {
      return true;
    }

    // Extract required leader types from card text
    // Patterns to look for:
    // - "if your leader has the {X} type"
    // - "if your leader's type includes \"X\""
    // - "{X} type" in conditional contexts
    const leaderSubtypes = new Set(
      (leader.subtypes ?? []).map((s) => s.toLowerCase().trim()).filter(Boolean),
    );
    const leaderName = (leader.name ?? '').toLowerCase();

    // Check for conditional requirements
    const conditionalPatterns = [
      // Pattern: "if your leader has the {X} type" or "If your Leader has the {X} type"
      /if\s+your\s+leader\s+has\s+the\s+\{([^}]+)\}\s+type/gi,
      // Pattern: "if your leader's type includes \"X\""
      /if\s+your\s+leader'?s?\s+type\s+includes\s+["']([^"']+)["']/gi,
      // Pattern: "if your leader's type includes {X}"
      /if\s+your\s+leader'?s?\s+type\s+includes\s+\{([^}]+)\}/gi,
      // Pattern: "if your leader is {X} type"
      /if\s+your\s+leader\s+is\s+\{([^}]+)\}\s+type/gi,
    ];

    const requiredTypes = new Set<string>();

    for (const pattern of conditionalPatterns) {
      let match;
      pattern.lastIndex = 0; // Reset regex
      while ((match = pattern.exec(cardText)) !== null) {
        const requiredType = match[1].toLowerCase().trim();
        if (requiredType) {
          requiredTypes.add(requiredType);
        }
      }
    }

    // If no conditional requirements found, card can be used
    if (requiredTypes.size === 0) {
      return true;
    }

    // Check if leader matches any of the required types
    for (const requiredType of requiredTypes) {
      // Check if leader has this subtype
      if (leaderSubtypes.has(requiredType)) {
        return true;
      }
      
      // Check if leader name contains this type (for cases like "Straw Hat Crew" in leader name)
      if (leaderName.includes(requiredType)) {
        return true;
      }
      
      // Check if any leader subtype contains the required type (for partial matches)
      for (const subtype of leaderSubtypes) {
        if (subtype.includes(requiredType) || requiredType.includes(subtype)) {
          return true;
        }
      }
    }

    // If we found conditional requirements but leader doesn't match any, card cannot be used
    return false;
  }

  /**
   * Checks if a card requires specific character types to exist in the deck (not just the leader).
   * Examples:
   * - "Play up to 1 {Punk Hazard} type Character card" - requires Punk Hazard characters in deck
   * - "Search your deck for a {Straw Hat Crew} type Character" - requires Straw Hat Crew in deck
   * - "If you have 5 or more {Navy} type Characters" - requires Navy characters in deck
   * 
   * @param card The card to check
   * @param existingCards The cards already in the deck
   * @param leader The leader card (some effects work if leader has the type)
   * @returns true if the card can be used (either no deck requirements, or required types exist)
   */
  private canUseCardWithDeckRequirements(
    card: import('@shared/types/optcg-card').OptcgCard,
    existingCards: DeckCardSuggestionWithCard[],
    leader?: import('@shared/types/optcg-card').OptcgCard | null,
  ): boolean {
    // Normalize text: remove HTML tags, convert to lowercase
    let cardText = (card.text ?? '').toLowerCase();
    // Remove HTML tags like <br>, <b>, etc.
    cardText = cardText.replace(/<[^>]+>/g, ' ');
    // Normalize whitespace
    cardText = cardText.replace(/\s+/g, ' ').trim();
    
    // If card has no text, it can always be used
    if (!cardText || cardText.length === 0) {
      return true;
    }

    // Patterns that require specific types in the deck (not just leader)
    const deckRequirementPatterns = [
      // "Play up to 1 {Punk Hazard} type Character card"
      /play\s+up\s+to\s+\d+\s+\{([^}]+)\}\s+type/gi,
      // "Search your deck for a {Straw Hat Crew} type Character"
      /search\s+your\s+deck\s+for\s+(?:a|an|up\s+to\s+\d+)\s+\{([^}]+)\}\s+type/gi,
      // "Add up to 1 {X} type Character" or "Add up to 1 {X} type card"
      /add\s+(?:up\s+to\s+)?\d+\s+\{([^}]+)\}\s+type/gi,
      // "If you have 5 or more {Navy} type Characters"
      /if\s+you\s+have\s+\d+\s+or\s+more\s+\{([^}]+)\}\s+type/gi,
      // "When you play a {X} type Character"
      /when\s+you\s+play\s+a\s+\{([^}]+)\}\s+type/gi,
      // "If you control a {X} type Character"
      /if\s+you\s+control\s+a\s+\{([^}]+)\}\s+type/gi,
      // "Look at up to 1 {X} type Character" or "Look at up to 1 {X} type card"
      /look\s+at\s+up\s+to\s+\d+\s+\{([^}]+)\}\s+type/gi,
      // "Reveal up to 1 {FILM} type card" or "Reveal up to 1 {X} type Character"
      /reveal\s+up\s+to\s+\d+\s+\{([^}]+)\}\s+type/gi,
      // "Reveal 1 {X} type card" (without "up to")
      /reveal\s+\d+\s+\{([^}]+)\}\s+type/gi,
      // "Search your deck for 1 {X} type card"
      /search\s+your\s+deck\s+for\s+\d+\s+\{([^}]+)\}\s+type/gi,
      // "Add 1 {X} type card" (without "up to")
      /add\s+\d+\s+\{([^}]+)\}\s+type/gi,
    ];

    // Check for event requirements (e.g., "trash an Event", "reveal 2 Events", "when you trash an Event")
    // Patterns: "trash an Event", "reveal 2 Events", "reveal up to 1 Event", etc.
    // Note: Allow text between number and "event" (e.g., "reveal up to 1 [Monkey.D.Luffy] or red Event")
    // Use a more flexible pattern that matches "reveal X ... event" where X is a number/word and ... can be anything
    const requiresEvent = /(?:trash|trashed|reveal|revealed|when\s+you\s+trash|if\s+you\s+trash|you\s+may\s+reveal)\s+(?:an|a|\d+|up\s+to\s+\d+).*?event/gi.test(cardText);
    
    // Color-specific events: "trash a {Red} Event", "reveal up to 1 {Red} Event", "reveal up to 1 [Monkey.D.Luffy] or red Event"
    // Try to extract color from patterns like "red Event", "{Red} Event", "[Red] Event"
    // First check for the "or" pattern: "reveal up to 1 [Monkey.D.Luffy] or red Event"
    const orPatternMatch = /reveal\s+(?:up\s+to\s+)?\d+\s+\[([^\]]+)\]\s+or\s+(\w+)\s+event/gi.exec(cardText);
    let requiredEventColor: string | null = null;
    if (orPatternMatch && orPatternMatch[2]) {
      requiredEventColor = orPatternMatch[2].toLowerCase().trim();
    } else {
      // Try other color patterns: "{Red} Event", "[Red] Event", "red Event"
      const colorEventMatch = /(?:trash|trashed|reveal|revealed|when\s+you\s+trash|if\s+you\s+trash|you\s+may\s+reveal).*?(?:\{([^}]+)\}|\[([^\]]+)\]|(\w+))\s+event/gi.exec(cardText);
      requiredEventColor = colorEventMatch ? (colorEventMatch[1] || colorEventMatch[2] || colorEventMatch[3])?.toLowerCase().trim() : null;
    }

    // Patterns that require specific character names in the deck
    // Examples: "If you have a {Buggy} or {Mohji} Character"
    const characterNameRequirementPatterns = [
      // "If you have a {Buggy} or {Mohji} Character"
      /if\s+you\s+have\s+(?:a|an)\s+\{([^}]+)\}(?:\s+or\s+\{([^}]+)\})*\s+character/gi,
      // "If you control a {Buggy} Character"
      /if\s+you\s+control\s+(?:a|an)\s+\{([^}]+)\}(?:\s+or\s+\{([^}]+)\})*\s+character/gi,
      // "When you play a {Buggy} Character"
      /when\s+you\s+play\s+(?:a|an)\s+\{([^}]+)\}(?:\s+or\s+\{([^}]+)\})*\s+character/gi,
      // "Search your deck for a {Buggy} Character"
      /search\s+your\s+deck\s+for\s+(?:a|an)\s+\{([^}]+)\}(?:\s+or\s+\{([^}]+)\})*\s+character/gi,
    ];

    const requiredTypes = new Set<string>();
    const requiredCharacterNames = new Set<string>();

    for (const pattern of deckRequirementPatterns) {
      let match;
      // Reset regex lastIndex to avoid issues with global regex
      pattern.lastIndex = 0;
      while ((match = pattern.exec(cardText)) !== null) {
        const requiredType = match[1].toLowerCase().trim();
        if (requiredType) {
          requiredTypes.add(requiredType);
        }
      }
    }

    // Extract character name requirements (e.g., {Buggy}, {Mohji})
    for (const pattern of characterNameRequirementPatterns) {
      let match;
      pattern.lastIndex = 0;
      while ((match = pattern.exec(cardText)) !== null) {
        // First capture group is the first character name
        if (match[1]) {
          requiredCharacterNames.add(match[1].toLowerCase().trim());
        }
        // Second capture group (if exists) is additional character name (e.g., "or {Mohji}")
        if (match[2]) {
          requiredCharacterNames.add(match[2].toLowerCase().trim());
        }
        // Handle cases with more than 2 names (though pattern may need adjustment)
        for (let i = 3; i < match.length; i++) {
          if (match[i]) {
            requiredCharacterNames.add(match[i].toLowerCase().trim());
          }
        }
      }
    }

    // Also check for standalone character name patterns like "{Buggy}" or "{Mohji}" in conditional contexts
    // This catches cases like "If you have a {Buggy} or {Mohji} Character" that might not match the above patterns
    // Also handles character names with dots like "{Monkey.D.Luffy}" or "{Monkey D. Luffy}"
    const standaloneCharacterPattern = /\{([A-Za-z\s\.]+)\}(?:\s+or\s+\{([A-Za-z\s\.]+)\})*(?:\s+character)?/gi;
    let standaloneMatch;
    standaloneCharacterPattern.lastIndex = 0;
    while ((standaloneMatch = standaloneCharacterPattern.exec(cardText)) !== null) {
      // Check if this appears in a conditional context (if/when you have/control/play)
      const beforeMatch = cardText.substring(Math.max(0, standaloneMatch.index - 50), standaloneMatch.index);
      if (/if\s+you\s+(?:have|control|play)/i.test(beforeMatch) || /when\s+you\s+(?:have|control|play)/i.test(beforeMatch)) {
        if (standaloneMatch[1]) {
          // Normalize character names (handle dots and spaces)
          const normalized = standaloneMatch[1].toLowerCase().trim().replace(/\s+/g, ' ').replace(/\./g, '.');
          requiredCharacterNames.add(normalized);
          // Also add without dots for matching flexibility
          requiredCharacterNames.add(normalized.replace(/\./g, ''));
        }
        if (standaloneMatch[2]) {
          const normalized = standaloneMatch[2].toLowerCase().trim().replace(/\s+/g, ' ').replace(/\./g, '.');
          requiredCharacterNames.add(normalized);
          requiredCharacterNames.add(normalized.replace(/\./g, ''));
        }
      }
    }
    
    // Check for character name requirements with "or" patterns (e.g., "{Monkey.D.Luffy} or {Red} Event" or "[Monkey.D.Luffy] or red Event")
    // Handle both curly braces {X} and square brackets [X] for character names
    // Pattern: "[Monkey.D.Luffy] or red Event" or "{Monkey.D.Luffy} or {Red} Event"
    const characterOrEventPattern = /(?:\{([^}]+)\}|\[([^\]]+)\])\s+or\s+(?:\{([^}]+)\}|\[([^\]]+)\]|(\w+))\s+event/gi;
    let charOrEventMatch;
    characterOrEventPattern.lastIndex = 0;
    while ((charOrEventMatch = characterOrEventPattern.exec(cardText)) !== null) {
      // First capture group is character name (from {X} or [X])
      const charName = charOrEventMatch[1] || charOrEventMatch[2];
      if (charName) {
        const normalized = charName.toLowerCase().trim().replace(/\s+/g, ' ').replace(/\./g, '.');
        requiredCharacterNames.add(normalized);
        requiredCharacterNames.add(normalized.replace(/\./g, ''));
      }
      // Second capture group is color (from {X}, [X], or plain word like "red")
      const colorName = charOrEventMatch[3] || charOrEventMatch[4] || charOrEventMatch[5];
      if (colorName) {
        // This is a color requirement for events
        requiredEventColor = colorName.toLowerCase().trim();
      }
    }
    
    // Also check for pattern: "reveal up to 1 [Monkey.D.Luffy] or red Event" (character name in brackets, color as plain word)
    const revealCharOrEventPattern = /reveal\s+(?:up\s+to\s+)?\d+\s+\[([^\]]+)\]\s+or\s+(\w+)\s+event/gi;
    let revealMatch;
    revealCharOrEventPattern.lastIndex = 0;
    while ((revealMatch = revealCharOrEventPattern.exec(cardText)) !== null) {
      if (revealMatch[1]) {
        const normalized = revealMatch[1].toLowerCase().trim().replace(/\s+/g, ' ').replace(/\./g, '.');
        requiredCharacterNames.add(normalized);
        requiredCharacterNames.add(normalized.replace(/\./g, ''));
      }
      if (revealMatch[2]) {
        requiredEventColor = revealMatch[2].toLowerCase().trim();
      }
    }

    // Check event requirements first (before checking other requirements)
    if (requiresEvent || requiredEventColor) {
      const hasEvents = existingCards.some((c) => {
        const cardType = (c.card.type ?? '').toLowerCase();
        return cardType.includes('event');
      });
      
      if (!hasEvents) {
        // Card requires events but deck has none
        this.logger.debug(
          `Card ${card.id} (${card.name}) requires events but deck has none. Card text: ${cardText.substring(0, 100)}`,
        );
        return false;
      }
      
      // If color-specific event required, check if deck has that color event
      if (requiredEventColor) {
        const hasColorEvent = existingCards.some((c) => {
          const cardType = (c.card.type ?? '').toLowerCase();
          const cardColor = (c.card.color ?? '').toLowerCase();
          return cardType.includes('event') && cardColor.includes(requiredEventColor);
        });
        
        if (!hasColorEvent) {
          // Check if required character name exists (alternative requirement)
          // For example: "{Monkey.D.Luffy} or {Red} Event" - if no red events, check for Luffy
          if (requiredCharacterNames.size === 0) {
            this.logger.debug(
              `Card ${card.id} (${card.name}) requires ${requiredEventColor} events but deck has none. Card text: ${cardText.substring(0, 100)}`,
            );
            return false; // No alternative requirement, card is unusable
          }
          // Will check character names below
        }
      }
    }

    // If no deck requirements found, card can be used
    if (requiredTypes.size === 0 && requiredCharacterNames.size === 0 && !requiresEvent) {
      return true;
    }

    // Build set of subtypes and character names that exist in the deck
    const existingSubtypes = new Set<string>();
    const existingCharacterNames = new Set<string>();
    const existingEventColors = new Set<string>();
    
    for (const existingCard of existingCards) {
      const subtypes = (existingCard.card.subtypes ?? [])
        .map((s) => s.toLowerCase().trim())
        .filter(Boolean);
      subtypes.forEach((s) => existingSubtypes.add(s));
      
      // Extract character name from card name (first word or key name)
      const cardName = (existingCard.card.name ?? '').toLowerCase().trim();
      if (cardName) {
        // Add full name
        existingCharacterNames.add(cardName);
        // Add first word (often the character name)
        const firstWord = cardName.split(/\s+/)[0];
        if (firstWord) {
          existingCharacterNames.add(firstWord);
        }
        // Add normalized versions (handle dots and spaces)
        const normalized = cardName.replace(/\s+/g, ' ').replace(/\./g, '.');
        existingCharacterNames.add(normalized);
        existingCharacterNames.add(normalized.replace(/\./g, ''));
        // Add name with dots normalized (e.g., "monkey.d.luffy" and "monkey d luffy")
        const withDots = cardName.replace(/\s+/g, '.').replace(/\.+/g, '.');
        existingCharacterNames.add(withDots);
      }
      
      // Track event colors for color-specific event requirements
      const cardType = (existingCard.card.type ?? '').toLowerCase();
      if (cardType.includes('event')) {
        const cardColor = (existingCard.card.color ?? '').toLowerCase();
        if (cardColor) {
          existingEventColors.add(cardColor);
          // Also add individual color words (e.g., "red" from "red/blue")
          cardColor.split(/[\/,\s]+/).forEach((c) => {
            if (c.trim()) {
              existingEventColors.add(c.trim());
            }
          });
        }
      }
    }

    // Check if the card itself has any of the required types
    // (e.g., if card says "reveal up to 1 {FILM} type card" and card itself is FILM type)
    const cardSubtypes = new Set(
      (card.subtypes ?? []).map((s) => s.toLowerCase().trim()).filter(Boolean),
    );
    for (const requiredType of requiredTypes) {
      if (cardSubtypes.has(requiredType)) {
        // Card itself has the required type, so it can satisfy its own requirement
        return true;
      }
    }

    // Also check if leader has any of the required types (some cards work if leader has it)
    if (leader) {
      const leaderSubtypes = new Set(
        (leader.subtypes ?? []).map((s) => s.toLowerCase().trim()).filter(Boolean),
      );
      for (const requiredType of requiredTypes) {
        if (leaderSubtypes.has(requiredType)) {
          // Leader has the type, so the card can potentially be used
          // But for "Play up to 1 {X} type Character", we still need it in deck
          // For now, we'll be lenient and allow it if leader has the type
          return true;
        }
      }
    }

    // Check if any required type exists in the deck
    for (const requiredType of requiredTypes) {
      if (existingSubtypes.has(requiredType)) {
        return true; // At least one required type exists in deck
      }
      
      // Check for partial matches (e.g., "Straw Hat Crew" matches "Straw Hat")
      for (const existingSubtype of existingSubtypes) {
        if (existingSubtype.includes(requiredType) || requiredType.includes(existingSubtype)) {
          return true;
        }
      }
    }

    // Check if any required character name exists in the deck
    for (const requiredName of requiredCharacterNames) {
      // Check exact match
      if (existingCharacterNames.has(requiredName)) {
        return true;
      }
      
      // Check normalized matches (handle dots and spaces variations)
      const normalizedRequired = requiredName.replace(/\./g, '').replace(/\s+/g, '');
      for (const existingName of existingCharacterNames) {
        const normalizedExisting = existingName.replace(/\./g, '').replace(/\s+/g, '');
        if (normalizedExisting === normalizedRequired) {
          return true;
        }
      }
      
      // Check if any existing character name contains the required name (or vice versa)
      for (const existingName of existingCharacterNames) {
        if (existingName.includes(requiredName) || requiredName.includes(existingName)) {
          return true;
        }
        // Also check normalized versions
        const normalizedExisting = existingName.replace(/\./g, '').replace(/\s+/g, '');
        const normalizedReq = requiredName.replace(/\./g, '').replace(/\s+/g, '');
        if (normalizedExisting.includes(normalizedReq) || normalizedReq.includes(normalizedExisting)) {
          return true;
        }
      }
    }
    
    // If we have event requirement but no character name match, check if color event exists
    if (requiredEventColor && requiredCharacterNames.size > 0) {
      // This is an "or" requirement: character OR color event
      if (existingEventColors.has(requiredEventColor)) {
        return true; // Color event exists, requirement satisfied
      }
    }

    // If we found deck requirements but none exist in deck (or leader), card is unusable
    return false;
  }

  /**
   * Checks if a card's conditional ability can be triggered with the given leader.
   * Filters out cards with abilities that can never activate (e.g., "if your Leader has 0 power or less" with a 5000 power leader).
   * 
   * @param card The card to check
   * @param leader The leader card
   * @returns true if the card's conditional abilities can potentially trigger, false if they can never trigger
   */
  private canCardAbilityTriggerWithLeader(
    card: import('@shared/types/optcg-card').OptcgCard,
    leader?: import('@shared/types/optcg-card').OptcgCard | null,
  ): boolean {
    if (!leader) {
      return true; // Can't validate without leader
    }

    // Normalize text: remove HTML tags, convert to lowercase
    // Check both card.text and card.raw?.ability to ensure we get the ability text
    let cardText = (card.text ?? '').toLowerCase();
    if (!cardText && (card as any).raw?.ability) {
      cardText = ((card as any).raw.ability ?? '').toLowerCase();
    }
    cardText = cardText.replace(/<[^>]+>/g, ' ');
    cardText = cardText.replace(/\s+/g, ' ').trim();
    
    if (!cardText || cardText.length === 0) {
      return true; // No conditional abilities
    }

    const leaderPower = leader.power ? parseInt(leader.power, 10) : null;
    if (leaderPower === null) {
      return true; // Can't validate without leader power
    }

    // Pattern: "if your Leader has 0 power or less"
    // Pattern: "if your Leader has X power or less"
    const leaderPowerOrLessPattern = /if\s+your\s+leader\s+has\s+(\d+)\s+power\s+or\s+less/gi;
    let match;
    leaderPowerOrLessPattern.lastIndex = 0;
    while ((match = leaderPowerOrLessPattern.exec(cardText)) !== null) {
      const requiredPower = parseInt(match[1], 10);
      if (leaderPower > requiredPower) {
        this.logger.warn(
          `Card ${card.id} (${card.name}) requires leader to have ${requiredPower} power or less, but leader has ${leaderPower} power. Ability cannot trigger. Card text: "${cardText.substring(0, 100)}"`,
        );
        return false; // Leader power is too high, ability can never trigger
      }
    }

    // Pattern: "if your Leader has X power or more"
    const leaderPowerOrMorePattern = /if\s+your\s+leader\s+has\s+(\d+)\s+power\s+or\s+more/gi;
    leaderPowerOrMorePattern.lastIndex = 0;
    while ((match = leaderPowerOrMorePattern.exec(cardText)) !== null) {
      const requiredPower = parseInt(match[1], 10);
      if (leaderPower < requiredPower) {
        this.logger.warn(
          `Card ${card.id} (${card.name}) requires leader to have ${requiredPower} power or more, but leader has ${leaderPower} power. Ability cannot trigger. Card text: "${cardText.substring(0, 100)}"`,
        );
        return false; // Leader power is too low, ability can never trigger
      }
    }

    // Pattern: "if your Leader has exactly X power"
    const leaderPowerExactPattern = /if\s+your\s+leader\s+has\s+exactly\s+(\d+)\s+power/gi;
    leaderPowerExactPattern.lastIndex = 0;
    while ((match = leaderPowerExactPattern.exec(cardText)) !== null) {
      const requiredPower = parseInt(match[1], 10);
      if (leaderPower !== requiredPower) {
        this.logger.warn(
          `Card ${card.id} (${card.name}) requires leader to have exactly ${requiredPower} power, but leader has ${leaderPower} power. Ability cannot trigger. Card text: "${cardText.substring(0, 100)}"`,
        );
        return false; // Leader power doesn't match exactly
      }
    }

    // Check for leader type/subtype requirements
    const leaderSubtypes = new Set(
      (leader.subtypes ?? []).map((s) => s.toLowerCase().trim()).filter(Boolean),
    );
    const leaderName = (leader.name ?? '').toLowerCase().trim();

    // Pattern: "if your Leader has the {Type} type"
    // Example: "if your Leader has the {Land of Wano} type"
    const leaderTypePattern = /if\s+your\s+leader\s+has\s+the\s+\{([^}]+)\}\s+type/gi;
    leaderTypePattern.lastIndex = 0;
    while ((match = leaderTypePattern.exec(cardText)) !== null) {
      const requiredType = match[1].toLowerCase().trim();
      if (!leaderSubtypes.has(requiredType)) {
        this.logger.warn(
          `Card ${card.id} (${card.name}) requires leader to have the {${match[1]}} type, but leader has types: [${Array.from(leaderSubtypes).join(', ')}]. Ability cannot trigger. Card text: "${cardText.substring(0, 100)}"`,
        );
        return false; // Leader doesn't have required type
      }
    }

    // Pattern: "if your Leader is [CharacterName]"
    // Example: "if your Leader is [Portgas.D.Ace]"
    const leaderNamePattern = /if\s+your\s+leader\s+is\s+\[([^\]]+)\]/gi;
    leaderNamePattern.lastIndex = 0;
    while ((match = leaderNamePattern.exec(cardText)) !== null) {
      const requiredName = match[1].toLowerCase().trim();
      // Normalize name variations (Monkey.D.Luffy vs Monkey D. Luffy)
      const normalizedRequired = requiredName.replace(/\./g, ' ').replace(/\s+/g, ' ').trim();
      const normalizedLeader = leaderName.replace(/\./g, ' ').replace(/\s+/g, ' ').trim();
      if (normalizedLeader !== normalizedRequired) {
        this.logger.warn(
          `Card ${card.id} (${card.name}) requires leader to be [${match[1]}], but leader is ${leader.name}. Ability cannot trigger. Card text: "${cardText.substring(0, 100)}"`,
        );
        return false; // Leader name doesn't match
      }
    }

    // Pattern: "if your Leader has the {Type} type or is [CharacterName]"
    // Example: "if your Leader has the {Land of Wano} type or is [Portgas.D.Ace]"
    const leaderTypeOrNamePattern = /if\s+your\s+leader\s+has\s+the\s+\{([^}]+)\}\s+type\s+or\s+is\s+\[([^\]]+)\]/gi;
    leaderTypeOrNamePattern.lastIndex = 0;
    while ((match = leaderTypeOrNamePattern.exec(cardText)) !== null) {
      const requiredType = match[1].toLowerCase().trim();
      const requiredName = match[2].toLowerCase().trim();
      const normalizedRequired = requiredName.replace(/\./g, ' ').replace(/\s+/g, ' ').trim();
      const normalizedLeader = leaderName.replace(/\./g, ' ').replace(/\s+/g, ' ').trim();
      
      const hasType = leaderSubtypes.has(requiredType);
      const isName = normalizedLeader === normalizedRequired;
      
      if (!hasType && !isName) {
        this.logger.warn(
          `Card ${card.id} (${card.name}) requires leader to have the {${match[1]}} type or be [${match[2]}], but leader has types: [${Array.from(leaderSubtypes).join(', ')}] and name: ${leader.name}. Ability cannot trigger. Card text: "${cardText.substring(0, 100)}"`,
        );
        return false; // Leader doesn't meet either requirement
      }
    }

    return true; // All conditional abilities can potentially trigger
  }

  private async hasTournamentSynergy(
    cardId: string,
    leaderId: string,
    existingCardIds: string[],
  ): Promise<boolean> {
    if (!leaderId) {
      return false;
    }

    const normalizedLeader = leaderId.trim().toUpperCase();
    const normalizedCard = cardId.trim().toUpperCase();
    const normalizedExisting = new Set(
      existingCardIds.map((id) => id.trim().toUpperCase()).filter(Boolean),
    );

    try {
      const leaderFrequency =
        await this.tournamentSynergy.getLeaderCardFrequency(normalizedLeader);
      if (leaderFrequency && leaderFrequency[normalizedCard] && leaderFrequency[normalizedCard] > 0) {
        return true;
      }
    } catch (error) {
      this.logger.debug(
        `Tournament synergy leader frequency lookup failed for ${normalizedCard} with leader ${normalizedLeader}: ${
          (error as Error).message
        }`,
      );
    }

    try {
      const synergyCandidates = await this.tournamentSynergy.getCardSynergies(
        normalizedCard,
        normalizedLeader,
        25,
      );
      if (
        synergyCandidates.some((candidate) => normalizedExisting.has(candidate.trim().toUpperCase()))
      ) {
        return true;
      }
    } catch (error) {
      this.logger.debug(
        `Tournament synergy co-occurrence lookup failed for ${normalizedCard} with leader ${normalizedLeader}: ${
          (error as Error).message
        }`,
      );
    }

    if (normalizedExisting.size > 0) {
      try {
        const globalSynergy = await this.tournamentSynergy.getCardSynergies(normalizedCard, undefined, 25);
        if (globalSynergy.some((candidate) => normalizedExisting.has(candidate.trim().toUpperCase()))) {
          return true;
        }
      } catch (error) {
        this.logger.debug(
          `Tournament global synergy lookup failed for ${normalizedCard}: ${(error as Error).message}`,
        );
      }
    }

    return false;
  }

  private supportsLeaderAbility(
    card: import('@shared/types/optcg-card').OptcgCard,
    leader: import('@shared/types/optcg-card').OptcgCard,
  ): boolean {
    const leaderText = (leader.text ?? '').toLowerCase().trim();
    if (leaderText.length === 0) {
      return false;
    }

    const cardText = (card.text ?? '').toLowerCase();
    const cardName = (card.name ?? '').toLowerCase();
    const leaderName = (leader.name ?? '').toLowerCase();

    if (!cardText && !cardName) {
      return false;
    }

    const leaderNameToken = leaderName.split(' ')[0];
    if (leaderNameToken && (cardText.includes(leaderNameToken) || cardName.includes(leaderNameToken))) {
      return true;
    }

    for (const rule of LEADER_ABILITY_SYNERGY_RULES) {
      if (!rule.abilityPattern.test(leaderText)) {
        continue;
      }
      if (
        rule.supportKeywords.some(
          (keyword) => cardText.includes(keyword.toLowerCase()) || cardName.includes(keyword.toLowerCase()),
        )
      ) {
        return true;
      }
    }

    const leaderSubtypes = new Set(
      (leader.subtypes ?? []).map((subtype) => subtype.toLowerCase().trim()).filter(Boolean),
    );
    const cardSubtypes = new Set(
      (card.subtypes ?? []).map((subtype) => subtype.toLowerCase().trim()).filter(Boolean),
    );
    if (leaderSubtypes.size > 0 && Array.from(cardSubtypes).some((subtype) => leaderSubtypes.has(subtype))) {
      return true;
    }

    const abilitySubtypePattern = /\{([^}]+)\}/g;
    let match: RegExpExecArray | null;
    const abilityReferencedSubtypes = new Set<string>();
    while ((match = abilitySubtypePattern.exec(leaderText)) !== null) {
      const extracted = match[1].toLowerCase().trim();
      if (extracted) {
        abilityReferencedSubtypes.add(extracted);
      }
    }
    if (
      abilityReferencedSubtypes.size > 0 &&
      Array.from(cardSubtypes).some((subtype) => abilityReferencedSubtypes.has(subtype))
    ) {
      return true;
    }

    const bracketPattern = /\[([^\]]+)\]/g;
    while ((match = bracketPattern.exec(leaderText)) !== null) {
      const keyword = match[1].toLowerCase().trim();
      if (!keyword || keyword.length < 3) {
        continue;
      }
      if (cardText.includes(keyword) || cardName.includes(keyword)) {
        return true;
      }
    }

    return false;
  }

  private hasConflictsWithLeader(
    card: import('@shared/types/optcg-card').OptcgCard,
    leader: import('@shared/types/optcg-card').OptcgCard,
    leaderColors: Set<string>,
    existingCards?: DeckCardSuggestionWithCard[],
  ): boolean {
    if (!this.isColorLegal(card.color ?? null, leaderColors)) {
      return true;
    }

    if (!this.canUseCardEffects(card, leader)) {
      return true;
    }

    // Check if card requires deck-wide type requirements that don't exist
    if (existingCards && !this.canUseCardWithDeckRequirements(card, existingCards, leader)) {
      return true;
    }

    const cardText = (card.text ?? '').toLowerCase();
    if (!cardText) {
      return false;
    }

    const leaderSubtypes = new Set(
      (leader.subtypes ?? []).map((subtype) => subtype.toLowerCase().trim()).filter(Boolean),
    );

    const negativeRequirementPatterns = [
      /if\s+your\s+leader\s+does\s+not\s+have\s+the\s+\{([^}]+)\}\s+type/gi,
      /if\s+your\s+leader\s+is\s+not\s+\{([^}]+)\}\s+type/gi,
      /unless\s+your\s+leader'?s?\s+type\s+includes\s+\{([^}]+)\}/gi,
    ];

    for (const pattern of negativeRequirementPatterns) {
      let match: RegExpExecArray | null;
      pattern.lastIndex = 0; // Reset regex
      while ((match = pattern.exec(cardText)) !== null) {
        const requiredType = match[1].toLowerCase().trim();
        if (requiredType && !leaderSubtypes.has(requiredType)) {
          return true;
        }
      }
    }

    const exclusionPatterns = [
      /if\s+your\s+leader\s+is\s+\{([^}]+)\}\s+type[^.]*cannot/gi,
      /if\s+your\s+leader\s+has\s+the\s+\{([^}]+)\}\s+type[^.]*cannot/gi,
    ];

    for (const pattern of exclusionPatterns) {
      let match: RegExpExecArray | null;
      pattern.lastIndex = 0; // Reset regex
      while ((match = pattern.exec(cardText)) !== null) {
        const excludedType = match[1].toLowerCase().trim();
        if (excludedType && leaderSubtypes.has(excludedType)) {
          return true;
        }
      }
    }

    return false;
  }

  private async buildColorFallbackCards(
    leaderColors: Set<string>,
    leaderCardId: string,
    existing?: DeckCardSuggestionWithCard[],
    allowedSubtypes?: Set<string>,
    leader?: import('@shared/types/optcg-card').OptcgCard | null,
    prompt?: string,
    allowVanilla: boolean = false,
    filters?: { allowedSetIds: Set<string>; metaOnly: boolean },
  ): Promise<DeckCardSuggestionWithCard[]> {
    const allowedSetIds = filters?.allowedSetIds ?? new Set<string>();
    const metaOnly = filters?.metaOnly ?? false;

    if (!leaderColors.size) {
      return [];
    }

    const existingIds = new Set((existing ?? []).map((c) => this.getBaseCardCode(c.card.id)));
    existingIds.add(this.getBaseCardCode(leaderCardId));

    const candidateIds = new Set<string>();
    for (const color of leaderColors) {
      const records = await this.cardKnowledge.getCardsByColor(color);
      for (const rec of records) {
        candidateIds.add(rec.id.toUpperCase());
      }
    }

    // Filter out cards that already exist (by base code)
    const filteredCandidateIds = Array.from(candidateIds).filter((id) => {
      const baseCode = this.getBaseCardCode(id);
      return !existingIds.has(baseCode);
    });
    if (!filteredCandidateIds.length) {
      return [];
    }

    // Shuffle candidate IDs to add randomness to deck generation
    const shuffledCandidateIds = Array.from(filteredCandidateIds);
    for (let i = shuffledCandidateIds.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffledCandidateIds[i], shuffledCandidateIds[j]] = [shuffledCandidateIds[j], shuffledCandidateIds[i]];
    }

    const cards = await this.optcgApiService.getCardsBySetIds(shuffledCandidateIds.slice(0, 200));
    const leaderCard = leader ? await this.optcgApiService.getCardBySetId(leaderCardId).catch(() => null) : null;
    const results: DeckCardSuggestionWithCard[] = [];
    for (const card of cards) {
      if (!card) continue;
      if (!this.cardMeetsFilters(card, allowedSetIds, metaOnly, leaderCard ?? undefined)) continue;
      if (card.type?.toLowerCase() === 'leader') continue;
      if (!this.isColorLegal(card.color ?? null, leaderColors)) continue;

      // Apply archetype alignment if subtypes are provided
      if (allowedSubtypes && !this.isArchetypeAligned(card, allowedSubtypes)) continue;

      if (leader && this.hasConflictsWithLeader(card, leader, leaderColors, existing ?? [])) {
        continue;
      }

      // Exclude vanilla cards unless explicitly allowed (last resort)
      if (!allowVanilla && this.isVanillaCard(card)) {
        continue;
      }

      // If leader is provided, card MUST match leader type/subtypes (unless it's a filler card scenario)
      if (leader) {
        const leaderSubtypes = new Set(
          (leader.subtypes ?? []).map((s) => s.toLowerCase().trim()).filter(Boolean),
        );
        const leaderFamily = (leader.raw as any)?.family as string | undefined;
        const leaderFamilyParts = leaderFamily
          ? leaderFamily
              .split(/[\/,&]/)
              .map((p) => p.trim().toLowerCase())
              .filter(Boolean)
          : [];
        let allLeaderTypes = new Set([
          ...Array.from(leaderSubtypes).map((s) => s.toLowerCase().trim()),
          ...leaderFamilyParts,
        ]);

        // Special case: if leader has "?" as subtype, extract types from ability text
        if (allLeaderTypes.has('?')) {
          const leaderText = (leader.text ?? '').toLowerCase();
          // Extract types mentioned in ability (e.g., {Celestial Dragons}, {Mary Geoise})
          const abilityTypePattern = /\{([^}]+)\}/g;
          let match: RegExpExecArray | null;
          abilityTypePattern.lastIndex = 0; // Reset regex
          while ((match = abilityTypePattern.exec(leaderText)) !== null) {
            const extractedType = match[1].toLowerCase().trim();
            if (extractedType && extractedType !== '?') {
              allLeaderTypes.add(extractedType);
            }
          }
          // Remove "?" from the set since it's not a real type
          allLeaderTypes.delete('?');
        }

        if (allLeaderTypes.size > 0) {
          const cardSubtypes = new Set(
            (card.subtypes ?? []).map((s) => s.toLowerCase().trim()).filter(Boolean),
          );
          const cardFamily = (card.raw as any)?.family as string | undefined;
          const cardFamilyParts = cardFamily
            ? cardFamily
                .split(/[\/,&]/)
                .map((p) => p.trim().toLowerCase())
                .filter(Boolean)
            : [];
          const allCardTypes = new Set([...Array.from(cardSubtypes), ...cardFamilyParts]);

          // Card must share at least one type with leader
          const matchesLeaderType = Array.from(allLeaderTypes).some((leaderType) =>
            allCardTypes.has(leaderType),
          );

          if (!matchesLeaderType) {
            continue; // Skip cards that don't match leader type
          }
        }
      }

      if (leader && prompt) {
        const contributes = await this.cardContributesToDeck(card, prompt, leader, existing ?? []);
        if (!contributes) {
          continue;
        }
      }

      // Assign higher initial quantities to improve consistency
      // Prefer 3 copies for characters/events, 2 for others
      let initialQty = 2;
      const cardRole = this.resolveRole(card);
      if (cardRole === 'character' || cardRole === 'event') {
        initialQty = 3;
      }
      
      // Generate appropriate rationale based on card contribution
      let rationale = '';
      if (allowVanilla && this.isVanillaCard(card)) {
        rationale = `x${initialQty}: Basic card included for deck consistency.`;
      } else {
        // Check if card has synergy with leader or deck
        const cardText = (card.text ?? '').toLowerCase();
        const leaderText = (leader?.text ?? '').toLowerCase();
        const leaderName = leader?.name?.toLowerCase().split(' ')[0] || '';
        const hasSynergy = leader && (
          cardText.includes(leaderName) ||
          (card.subtypes ?? []).some(subtype => 
            (leader.subtypes ?? []).some(leaderSubtype => 
              subtype.toLowerCase() === leaderSubtype.toLowerCase()
            )
          )
        );
        
        // Get leader type for rationale
        const leaderFamily = leader ? ((leader.raw as any)?.family as string | undefined) : undefined;
        const leaderTypeStr = leaderFamily || (leader?.subtypes ?? []).join('/') || 'archetype';
        
        if (hasSynergy) {
          rationale = `x${initialQty}: Synergistic card that supports ${leader?.name || 'the leader'}'s ${leaderTypeStr} strategy.`;
        } else {
          rationale = `x${initialQty}: Card matching ${leader?.name || 'the leader'}'s ${leaderTypeStr} type.`;
        }
      }
      
      results.push({
        cardSetId: card.id,
        quantity: initialQty,
        rationale,
        role: cardRole,
        card,
      });
    }

    return results;
  }

  /**
   * More lenient version of buildColorFallbackCards that skips conditional effects and contribution checks.
   * Used when we need to find cards but strict filtering is too restrictive.
   */
  private async buildColorFallbackCardsLenient(
    leaderColors: Set<string>,
    leaderCardId: string,
    existing?: DeckCardSuggestionWithCard[],
    allowedSubtypes?: Set<string>,
    leader?: import('@shared/types/optcg-card').OptcgCard | null,
    allowVanilla: boolean = false,
    filters?: { allowedSetIds: Set<string>; metaOnly: boolean },
  ): Promise<DeckCardSuggestionWithCard[]> {
    const allowedSetIds = filters?.allowedSetIds ?? new Set<string>();
    const metaOnly = filters?.metaOnly ?? false;

    if (!leaderColors.size) {
      return [];
    }

    const existingIds = new Set((existing ?? []).map((c) => this.getBaseCardCode(c.card.id)));
    existingIds.add(this.getBaseCardCode(leaderCardId));

    const candidateIds = new Set<string>();
    for (const color of leaderColors) {
      const records = await this.cardKnowledge.getCardsByColor(color);
      for (const rec of records) {
        candidateIds.add(rec.id.toUpperCase());
      }
    }
    
    this.logger.debug(
      `buildColorFallbackCardsLenient: Found ${candidateIds.size} candidate cards by color (${Array.from(leaderColors).join(', ')}). Set filter: ${allowedSetIds.size > 0 ? Array.from(allowedSetIds).join(', ') : 'none'}`,
    );

    // Filter out cards that already exist (by base code)
    let filteredCandidateIds = Array.from(candidateIds).filter((id) => {
      const baseCode = this.getBaseCardCode(id);
      return !existingIds.has(baseCode);
    });
    
    // Pre-filter by set code if set filter is active (before API call to reduce load)
    if (allowedSetIds.size > 0) {
      const beforeSetFilter = filteredCandidateIds.length;
      filteredCandidateIds = filteredCandidateIds.filter((id) => {
        const setCode = this.getSetCode(id);
        const normalizedSetCode = this.normalizeSetCode(setCode);
        return allowedSetIds.has(normalizedSetCode);
      });
      this.logger.debug(
        `Pre-filtered by set: ${filteredCandidateIds.length} cards (from ${beforeSetFilter} candidates)`,
      );
    }
    
    if (!filteredCandidateIds.length) {
      this.logger.debug(`No candidate cards found after filtering. Returning empty array.`);
      return [];
    }

    // Shuffle candidate IDs to add randomness to deck generation
    for (let i = filteredCandidateIds.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [filteredCandidateIds[i], filteredCandidateIds[j]] = [filteredCandidateIds[j], filteredCandidateIds[i]];
    }

    const cards = await this.optcgApiService.getCardsBySetIds(filteredCandidateIds.slice(0, 200));
    this.logger.debug(`Fetched ${cards.length} cards from API for fallback (requested ${Math.min(filteredCandidateIds.length, 200)})`);
    const leaderCard = leader ? await this.optcgApiService.getCardBySetId(leaderCardId).catch(() => null) : null;
    const results: DeckCardSuggestionWithCard[] = [];
    let filteredCount = 0;
    const filterReasons = { setMeta: 0, leader: 0, color: 0, archetype: 0, vanilla: 0 };
    
    for (const card of cards) {
      if (!card) continue;
      // Set filter already applied above, but double-check here
      if (!this.cardMeetsFilters(card, allowedSetIds, metaOnly, leaderCard ?? undefined)) {
        filterReasons.setMeta++;
        continue;
      }
      if (card.type?.toLowerCase() === 'leader') {
        filterReasons.leader++;
        continue;
      }
      if (!this.isColorLegal(card.color ?? null, leaderColors)) {
        filterReasons.color++;
        continue;
      }
      
      // Apply archetype alignment if subtypes are provided (but we're passing undefined when relaxing)
      if (allowedSubtypes && !this.isArchetypeAligned(card, allowedSubtypes)) {
        filterReasons.archetype++;
        continue;
      }
      
      // SKIP conditional effects check - too strict
      // SKIP contribution check - too strict
      
      // Check leader type matching, but be lenient for special cases like "?"
      if (leader) {
        const leaderSubtypes = new Set(
          (leader.subtypes ?? []).map((s) => s.toLowerCase().trim()).filter(Boolean),
        );
        const leaderFamily = (leader.raw as any)?.family as string | undefined;
        const leaderFamilyParts = leaderFamily
          ? leaderFamily
              .split(/[\/,&]/)
              .map((p) => p.trim().toLowerCase())
              .filter(Boolean)
          : [];
        let allLeaderTypes = new Set([
          ...Array.from(leaderSubtypes).map((s) => s.toLowerCase().trim()),
          ...leaderFamilyParts,
        ]);

        // Special case: if leader has "?" as subtype, extract types from ability text
        if (allLeaderTypes.has('?')) {
          const leaderText = (leader.text ?? '').toLowerCase();
          // Extract types mentioned in ability (e.g., {Celestial Dragons}, {Mary Geoise})
          const abilityTypePattern = /\{([^}]+)\}/g;
          let match: RegExpExecArray | null;
          abilityTypePattern.lastIndex = 0; // Reset regex
          while ((match = abilityTypePattern.exec(leaderText)) !== null) {
            const extractedType = match[1].toLowerCase().trim();
            if (extractedType && extractedType !== '?') {
              allLeaderTypes.add(extractedType);
            }
          }
          // Remove "?" from the set since it's not a real type
          allLeaderTypes.delete('?');
        }

        if (allLeaderTypes.size > 0) {
          const cardSubtypes = new Set(
            (card.subtypes ?? []).map((s) => s.toLowerCase().trim()).filter(Boolean),
          );
          const cardFamily = (card.raw as any)?.family as string | undefined;
          const cardFamilyParts = cardFamily
            ? cardFamily
                .split(/[\/,&]/)
                .map((p) => p.trim().toLowerCase())
                .filter(Boolean)
            : [];
          const allCardTypes = new Set([...Array.from(cardSubtypes), ...cardFamilyParts]);

          // Card must share at least one type with leader
          const matchesLeaderType = Array.from(allLeaderTypes).some((leaderType) =>
            allCardTypes.has(leaderType),
          );

          if (!matchesLeaderType) {
            filterReasons.archetype++;
            continue; // Skip cards that don't match leader type
          }
        }
      }
      
      // Exclude vanilla cards unless explicitly allowed (last resort)
      if (!allowVanilla && this.isVanillaCard(card)) {
        filterReasons.vanilla++;
        continue;
      }
      
      // Assign higher initial quantities to improve consistency
      // Prefer 3 copies for characters/events, 2 for others
      let initialQty = 2;
      const cardRole = this.resolveRole(card);
      if (cardRole === 'character' || cardRole === 'event') {
        initialQty = 3;
      }
      
      // Generate appropriate rationale based on card contribution
      let rationale = '';
      if (allowVanilla && this.isVanillaCard(card)) {
        rationale = `x${initialQty}: Basic card included for deck consistency.`;
      } else {
        // Check if card has synergy with leader or deck
        const cardText = (card.text ?? '').toLowerCase();
        const leaderText = (leader?.text ?? '').toLowerCase();
        const leaderName = leader?.name?.toLowerCase().split(' ')[0] || '';
        const hasSynergy = leader && (
          cardText.includes(leaderName) ||
          (card.subtypes ?? []).some(subtype => 
            (leader.subtypes ?? []).some(leaderSubtype => 
              subtype.toLowerCase() === leaderSubtype.toLowerCase()
            )
          )
        );
        
        // Get leader type for rationale
        const leaderFamily = leader ? ((leader.raw as any)?.family as string | undefined) : undefined;
        const leaderTypeStr = leaderFamily || (leader?.subtypes ?? []).join('/') || 'archetype';
        
        if (hasSynergy) {
          rationale = `x${initialQty}: Synergistic card that supports ${leader?.name || 'the leader'}'s ${leaderTypeStr} strategy.`;
        } else {
          rationale = `x${initialQty}: Card matching ${leader?.name || 'the leader'}'s ${leaderTypeStr} type.`;
        }
      }
      
      results.push({
        cardSetId: card.id,
        quantity: initialQty,
        rationale,
        role: cardRole,
        card,
      });
    }
    
    this.logger.debug(
      `buildColorFallbackCardsLenient: Added ${results.length} cards. ` +
      `Filtered out: setMeta=${filterReasons.setMeta}, leader=${filterReasons.leader}, ` +
      `color=${filterReasons.color}, archetype=${filterReasons.archetype}, vanilla=${filterReasons.vanilla}`,
    );
    
    return results;
  }

  private resolveRole(card?: import('@shared/types/optcg-card').OptcgCard | null): DeckCardSuggestionWithCard['role'] {
    const type = card?.type?.toLowerCase() ?? '';
    if (type.includes('character')) return 'character';
    if (type.includes('event')) return 'event';
    if (type.includes('stage')) return 'stage';
    if (type.includes('don')) return 'don';
    if (type.includes('counter')) return 'counter';
    return 'other';
  }

  /**
   * Ensures balanced color distribution for multi-color leaders.
   * If a leader has multiple colors, ensures cards from all colors are represented.
   */
  private async ensureColorBalance(
    cards: DeckCardSuggestionWithCard[],
    leaderColors: Set<string>,
    leaderCardId: string,
    leaderSubtypes: Set<string>,
    leader: import('@shared/types/optcg-card').OptcgCard,
    prompt: string,
    filters: { allowedSetIds: Set<string>; metaOnly: boolean },
  ): Promise<DeckCardSuggestionWithCard[]> {
    if (leaderColors.size <= 1) {
      return cards; // No need to balance if leader has only one color
    }

    // Count cards by color
    const colorCounts = new Map<string, number>();
    for (const color of leaderColors) {
      colorCounts.set(color, 0);
    }

    for (const card of cards) {
      const cardColors = this.parseColors(card.card.color);
      for (const color of cardColors) {
        if (colorCounts.has(color)) {
          colorCounts.set(color, colorCounts.get(color)! + card.quantity);
        }
      }
    }

    // Find the minimum count across all colors
    const minCount = Math.min(...Array.from(colorCounts.values()));
    const maxCount = Math.max(...Array.from(colorCounts.values()));
    const threshold = Math.max(5, Math.floor(maxCount * 0.3)); // At least 30% of max or 5 cards minimum

    // If any color has significantly fewer cards, add more of that color
    const result = cards.map((c) => ({ ...c }));
    const existingCardIds = new Set(result.map((c) => this.getBaseCardCode(c.card.id)));

    for (const [color, count] of colorCounts.entries()) {
      if (count < threshold && count < minCount + 5) {
        // This color is underrepresented, increase quantities of existing cards of this color
        const needed = Math.min(threshold - count, 10); // Add up to 10 cards worth
        
        // Find existing cards of this color
        const existingColorCards = result.filter((c) => {
          const cardColors = this.parseColors(c.card.color);
          return cardColors.includes(color);
        }).sort((a, b) => {
          // Prioritize: higher quantity first (to max out)
          return b.quantity - a.quantity;
        });

        let added = 0;
        for (const existing of existingColorCards) {
          if (added >= needed) break;
          const room = Math.min(4 - existing.quantity, needed - added);
          if (room > 0) {
            existing.quantity += room;
            added += room;
          }
        }
      }
    }

    return result;
  }

  private async enforceDeckRatios(
    cards: DeckCardSuggestionWithCard[],
    leaderColors: Set<string>,
    leaderCardId: string,
    leaderSubtypes: Set<string>,
    leader: import('@shared/types/optcg-card').OptcgCard,
    prompt: string,
    filters: { allowedSetIds: Set<string>; metaOnly: boolean },
  ): Promise<DeckCardSuggestionWithCard[]> {
    const result = cards.map((c) => ({ ...c }));
    const currentTotal = result.reduce((sum, c) => sum + c.quantity, 0);

    // Calculate current composition
    const characters = result.filter((c) => c.role === 'character');
    const events = result.filter((c) => c.role === 'event');
    const stages = result.filter((c) => c.role === 'stage');

    const charCount = characters.reduce((sum, c) => sum + c.quantity, 0);
    const eventCount = events.reduce((sum, c) => sum + c.quantity, 0);
    const stageCount = stages.reduce((sum, c) => sum + c.quantity, 0);

    // Target ratios (for 50-card deck)
    const targetChars = { min: 30, max: 35 }; // 60-70%
    const targetEvents = { min: 8, max: 12 }; // 15-25%
    const targetStages = { min: 0, max: 2 }; // 2-4% if needed

    // Build allowed subtypes from existing deck
    const deckSubtypes = new Set<string>();
    for (const card of result) {
      for (const subtype of card.card.subtypes ?? []) {
        deckSubtypes.add(subtype.toLowerCase().trim());
      }
    }
    const allowedSubtypes = new Set([...leaderSubtypes, ...deckSubtypes]);

    // Fill missing characters by increasing existing character quantities
    if (charCount < targetChars.min) {
      const needed = targetChars.min - charCount;
      // Only increase existing characters, don't add new ones
      const existingChars = characters.slice().sort((a, b) => {
        // Prioritize: higher quantity first (to max out), then by tier/importance
        return b.quantity - a.quantity;
      });
      
      let added = 0;
      for (const existing of existingChars) {
        if (added >= needed) break;
        const room = Math.min(4 - existing.quantity, needed - added);
        if (room > 0) {
          existing.quantity += room;
          added += room;
        }
      }
    }

    // Fill missing events by increasing existing event quantities
    const newEventCount = result.filter((c) => c.role === 'event').reduce((sum, c) => sum + c.quantity, 0);
    if (newEventCount < targetEvents.min) {
      const needed = targetEvents.min - newEventCount;
      // Only increase existing events, don't add new ones
      const existingEvents = events.slice().sort((a, b) => {
        // Prioritize: higher quantity first (to max out)
        return b.quantity - a.quantity;
      });
      
      let added = 0;
      for (const existing of existingEvents) {
        if (added >= needed) break;
        const room = Math.min(4 - existing.quantity, needed - added);
        if (room > 0) {
          existing.quantity += room;
          added += room;
        }
      }
    }

    // Add stages only if deck needs them (check if leader or key cards reference stages)
    const hasStageReference = result.some((c) => {
      const text = (c.card.text ?? '').toLowerCase();
      return text.includes('stage') || text.includes('moby') || text.includes('onigashima');
    });
    const newStageCount = result.filter((c) => c.role === 'stage').reduce((sum, c) => sum + c.quantity, 0);
    if (hasStageReference && newStageCount < targetStages.min) {
      const needed = targetStages.min - newStageCount;
      // Only increase existing stages, don't add new ones
      const existingStages = stages.slice().sort((a, b) => {
        // Prioritize: higher quantity first (to max out)
        return b.quantity - a.quantity;
      });
      
      let added = 0;
      for (const existing of existingStages) {
        if (added >= needed) break;
        const room = Math.min(4 - existing.quantity, needed - added);
        if (room > 0) {
          existing.quantity += room;
          added += room;
        }
      }
    }

    // Ensure we don't exceed max ratios - trim excess if needed
    const finalCharCount = result.filter((c) => c.role === 'character').reduce((sum, c) => sum + c.quantity, 0);
    const finalEventCount = result.filter((c) => c.role === 'event').reduce((sum, c) => sum + c.quantity, 0);
    const finalStageCount = result.filter((c) => c.role === 'stage').reduce((sum, c) => sum + c.quantity, 0);

    // If characters exceed max, reduce lower-priority characters
    if (finalCharCount > targetChars.max) {
      const excess = finalCharCount - targetChars.max;
      const sortedChars = result
        .filter((c) => c.role === 'character')
        .sort((a, b) => a.quantity - b.quantity || a.card.name.localeCompare(b.card.name));
      let reduced = 0;
      for (const card of sortedChars) {
        if (reduced >= excess) break;
        const reducible = Math.min(card.quantity - 1, excess - reduced);
        if (reducible > 0) {
          card.quantity -= reducible;
          reduced += reducible;
        }
      }
    }

    // If events exceed max, reduce lower-priority events
    if (finalEventCount > targetEvents.max) {
      const excess = finalEventCount - targetEvents.max;
      const sortedEvents = result
        .filter((c) => c.role === 'event')
        .sort((a, b) => a.quantity - b.quantity || a.card.name.localeCompare(b.card.name));
      let reduced = 0;
      for (const card of sortedEvents) {
        if (reduced >= excess) break;
        const reducible = Math.min(card.quantity - 1, excess - reduced);
        if (reducible > 0) {
          card.quantity -= reducible;
          reduced += reducible;
        }
      }
    }

    // Remove cards with 0 quantity
    return result.filter((c) => c.quantity > 0);
  }

  private buildGameplaySummary(
    prompt: string,
    leader: import('@shared/types/optcg-card').OptcgCard,
    cards: DeckCardSuggestionWithCard[],
  ): string {
    const trimmedPrompt = prompt.trim();
    
    // Analyze deck composition
    const characters = cards.filter((c) => c.role === 'character');
    const events = cards.filter((c) => c.role === 'event');
    const stages = cards.filter((c) => c.role === 'stage');
    
    const topCards = cards
      .slice()
      .sort((a, b) => b.quantity - a.quantity)
      .slice(0, 5)
      .map((card) => card.card.name);
    
    const highCostCards = characters.filter((c) => {
      const cost = parseInt(c.card.cost ?? '0', 10);
      return cost >= 5;
    });
    
    const lowCostCards = characters.filter((c) => {
      const cost = parseInt(c.card.cost ?? '0', 10);
      return cost <= 2;
    });
    
    const midCostCards = characters.filter((c) => {
      const cost = parseInt(c.card.cost ?? '0', 10);
      return cost >= 3 && cost <= 4;
    });
    
    // Determine deck archetype/strategy
    const isAggro = lowCostCards.length > highCostCards.length && characters.length > events.length;
    const isControl = events.length > 5 || highCostCards.length > lowCostCards.length;
    const isMidrange = !isAggro && !isControl;
    
    // Build gameplay description
    const sections: string[] = [];
    
    // Strategy overview
    let strategyType = 'midrange';
    if (isAggro) strategyType = 'aggressive';
    else if (isControl) strategyType = 'control';
    
    sections.push(
      `This ${strategyType} deck focuses on ${trimmedPrompt.toLowerCase()}${trimmedPrompt.endsWith('.') ? '' : '.'}`,
    );
    
    // Win Strategy
    const winStrategy = this.buildWinStrategy(leader, cards, isAggro, isControl, isMidrange, highCostCards, lowCostCards);
    sections.push(winStrategy);
    
    // Turn-by-turn scenarios
    const turnByTurn = this.buildTurnByTurnScenarios(
      leader,
      cards,
      lowCostCards,
      midCostCards,
      highCostCards,
      isAggro,
      isControl,
    );
    sections.push(turnByTurn);
    
    // Early game
    if (lowCostCards.length > 0) {
      const earlyGameCards = lowCostCards
        .slice(0, 3)
        .map((c) => c.card.name)
        .join(', ');
      sections.push(
        `Early game: Establish board presence with low-cost characters like ${earlyGameCards}.`,
      );
    }
    
    // Mid game
    if (midCostCards.length > 0) {
      const midGameCards = midCostCards
        .slice(0, 2)
        .map((c) => c.card.name)
        .join(', ');
      sections.push(
        `Mid game: Develop your strategy with ${midGameCards} and set up key synergies.`,
      );
    }
    
    // Late game / Win condition
    if (highCostCards.length > 0) {
      const lateGameCards = highCostCards
        .slice(0, 2)
        .map((c) => c.card.name)
        .join(', ');
      sections.push(
        `Late game: Close out with powerful finishers like ${lateGameCards} and ${leader.name}'s ability.`,
      );
    } else {
      sections.push(
        `Win condition: Maintain pressure with ${leader.name}'s ability and consistent board presence.`,
      );
    }
    
    // Key cards
    if (topCards.length > 0) {
      sections.push(`Key cards: ${topCards.slice(0, 3).join(', ')}.`);
    }
    
    return sections.join(' ');
  }

  private buildWinStrategy(
    leader: import('@shared/types/optcg-card').OptcgCard,
    cards: DeckCardSuggestionWithCard[],
    isAggro: boolean,
    isControl: boolean,
    isMidrange: boolean,
    highCostCards: DeckCardSuggestionWithCard[],
    lowCostCards: DeckCardSuggestionWithCard[],
  ): string {
    const leaderAbility = (leader.text ?? '').toLowerCase();
    const hasLifeRemoval = leaderAbility.includes('trash') && leaderAbility.includes('life');
    const hasBoardControl = leaderAbility.includes('k.o.') || leaderAbility.includes('rest');
    const hasCardAdvantage = leaderAbility.includes('draw') || leaderAbility.includes('add');
    
    if (isAggro) {
      if (hasLifeRemoval) {
        return `Win Strategy: Apply early pressure with low-cost characters, then use ${leader.name}'s life removal ability to finish the game. Focus on reducing opponent's life total quickly while maintaining board control.`;
      }
      return `Win Strategy: Overwhelm your opponent with early aggression. Play multiple characters each turn, attack with everything, and use ${leader.name}'s ability to maintain tempo. Win by reducing opponent's life to 0 before they can stabilize.`;
    }
    
    if (isControl) {
      if (hasBoardControl) {
        return `Win Strategy: Control the board early with removal and blockers. Use ${leader.name}'s ability to eliminate threats, then win with high-cost finishers once you've exhausted opponent's resources.`;
      }
      if (hasCardAdvantage) {
        return `Win Strategy: Outvalue your opponent through card advantage. Use ${leader.name}'s ability to draw cards and maintain resources. Win by having more options and better late-game threats.`;
      }
      return `Win Strategy: Survive the early game with blockers and events. Use ${leader.name}'s ability strategically to gain advantages. Win in the late game when you have more resources and better threats.`;
    }
    
    // Midrange
    if (highCostCards.length > 0) {
      return `Win Strategy: Establish board presence early, then transition to powerful late-game threats. Use ${leader.name}'s ability to support your game plan. Win by outvaluing opponents in the mid-to-late game.`;
    }
    
    return `Win Strategy: Maintain consistent pressure throughout the game. Use ${leader.name}'s ability to support your board and create favorable trades. Win by controlling the board and gradually reducing opponent's life.`;
  }

  private buildTurnByTurnScenarios(
    leader: import('@shared/types/optcg-card').OptcgCard,
    cards: DeckCardSuggestionWithCard[],
    lowCostCards: DeckCardSuggestionWithCard[],
    midCostCards: DeckCardSuggestionWithCard[],
    highCostCards: DeckCardSuggestionWithCard[],
    isAggro: boolean,
    isControl: boolean,
  ): string {
    const scenarios: string[] = [];
    
    // Turn 1-2 (Early)
    if (lowCostCards.length > 0) {
      const turn1Card = lowCostCards.find((c) => {
        const cost = parseInt(c.card.cost ?? '0', 10);
        return cost === 1;
      });
      const turn2Card = lowCostCards.find((c) => {
        const cost = parseInt(c.card.cost ?? '0', 10);
        return cost === 2;
      });
      
      if (isAggro) {
        scenarios.push(
          `Turns 1-2: Play ${turn1Card ? turn1Card.card.name : '1-cost characters'} on turn 1, ${turn2Card ? turn2Card.card.name : '2-cost characters'} on turn 2. Attack immediately to pressure opponent's life.`,
        );
      } else {
        scenarios.push(
          `Turns 1-2: Play ${turn1Card ? turn1Card.card.name : '1-cost characters'} for board presence. ${turn2Card ? `Use ${turn2Card.card.name} to` : 'Set up'} establish your engine or search for key pieces.`,
        );
      }
    }
    
    // Turn 3-4 (Mid)
    if (midCostCards.length > 0) {
      const turn3Card = midCostCards.find((c) => {
        const cost = parseInt(c.card.cost ?? '0', 10);
        return cost === 3;
      });
      const turn4Card = midCostCards.find((c) => {
        const cost = parseInt(c.card.cost ?? '0', 10);
        return cost === 4;
      });
      
      if (isAggro) {
        scenarios.push(
          `Turns 3-4: ${turn3Card ? `Play ${turn3Card.card.name} to` : 'Continue'} maintain pressure. ${turn4Card ? `${turn4Card.card.name} helps` : 'Use mid-cost characters to'} push for damage and set up lethal.`,
        );
      } else if (isControl) {
        scenarios.push(
          `Turns 3-4: ${turn3Card ? `Play ${turn3Card.card.name} for` : 'Establish'} board control. ${turn4Card ? `${turn4Card.card.name} provides` : 'Use blockers and removal to'} stabilize and prepare for late game.`,
        );
      } else {
        scenarios.push(
          `Turns 3-4: ${turn3Card ? `Play ${turn3Card.card.name} to` : 'Develop'} your board. ${turn4Card ? `${turn4Card.card.name} sets up` : 'Build towards'} your win condition while maintaining tempo.`,
        );
      }
    }
    
    // Turn 5+ (Late)
    if (highCostCards.length > 0) {
      const finisher = highCostCards[0];
      scenarios.push(
        `Turns 5+: Play ${finisher.card.name} as your finisher. Use ${leader.name}'s ability to support your game plan. Close out the game with powerful attacks or game-ending effects.`,
      );
    } else {
      scenarios.push(
        `Turns 5+: Continue applying pressure with your board. Use ${leader.name}'s ability each turn to maintain advantage. Win through consistent damage and resource management.`,
      );
    }
    
    return scenarios.join(' ');
  }

  private buildComboHighlights(
    prompt: string,
    leader: import('@shared/types/optcg-card').OptcgCard,
    cards: DeckCardSuggestionWithCard[],
  ): string[] {
    const combos: string[] = [];
    const characters = cards
      .filter((card) => card.role === 'character')
      .sort((a, b) => b.quantity - a.quantity)
      .slice(0, 3);
    const events = cards
      .filter((card) => card.role === 'event')
      .sort((a, b) => b.quantity - a.quantity)
      .slice(0, 2);

    characters.forEach((card) => {
      combos.push(
        `Use ${leader.name}'s ability alongside ${card.card.name} (${card.quantity}x) to pressure opponents as described in your prompt.`,
      );
    });

    events.forEach((card) => {
      combos.push(
        `Chain ${card.card.name} (${card.quantity}x) after ${leader.name} activates to maintain control and advance the game plan.`,
      );
    });

    if (!combos.length) {
      combos.push(`Follow the described game plan: ${prompt.trim()}.`);
    }

    return combos.slice(0, 4);
  }

  private buildBasicDeckReview(
    prompt: string,
    leader: import('@shared/types/optcg-card').OptcgCard,
    cards: DeckCardSuggestionWithCard[],
    gameplaySummary: string,
    comboHighlights: string[],
  ): string {
    const sections: string[] = [];
    
    // Analyze deck composition
    const characters = cards.filter((c) => c.role === 'character');
    const events = cards.filter((c) => c.role === 'event');
    const stages = cards.filter((c) => c.role === 'stage');
    const characterCount = characters.reduce((sum, c) => sum + c.quantity, 0);
    const eventCount = events.reduce((sum, c) => sum + c.quantity, 0);
    const stageCount = stages.reduce((sum, c) => sum + c.quantity, 0);
    
    const highCostCards = characters.filter((c) => {
      const cost = parseInt(c.card.cost ?? '0', 10);
      return cost >= 5;
    });
    const lowCostCards = characters.filter((c) => {
      const cost = parseInt(c.card.cost ?? '0', 10);
      return cost <= 2;
    });
    const midCostCards = characters.filter((c) => {
      const cost = parseInt(c.card.cost ?? '0', 10);
      return cost >= 3 && cost <= 4;
    });
    
    const isAggro = lowCostCards.length > highCostCards.length && characters.length > events.length;
    const isControl = events.length > 5 || highCostCards.length > lowCostCards.length;
    const strategyType = isAggro ? 'aggressive' : isControl ? 'control' : 'midrange';
    
    // 1. Overall Strategy
    sections.push('1. Overall Strategy');
    sections.push('');
    sections.push(`This ${strategyType} deck focuses on ${prompt.trim().toLowerCase()}${prompt.trim().endsWith('.') ? '' : '.'}`);
    sections.push('');
    sections.push(`The deck is built around ${leader.name}${leader.text ? `, whose ability enables ${leader.text.substring(0, 100)}...` : ' as the leader.'}`);
    sections.push('');
    
    // 2. Deck Composition
    sections.push('2. Deck Composition');
    sections.push('');
    sections.push(`- Characters: ${characterCount} cards (${lowCostCards.length} low-cost, ${midCostCards.length} mid-cost, ${highCostCards.length} high-cost)`);
    sections.push(`- Events: ${eventCount} cards`);
    if (stageCount > 0) {
      sections.push(`- Stages: ${stageCount} cards`);
    }
    sections.push('');
    
    // 3. Strengths and Weaknesses
    sections.push('3. Strengths and Weaknesses');
    sections.push('');
    sections.push('Strengths:');
    if (lowCostCards.length > 0) {
      sections.push(`- Strong early game presence with ${lowCostCards.length} low-cost characters`);
    }
    if (highCostCards.length > 0) {
      sections.push(`- Powerful late game finishers with ${highCostCards.length} high-cost characters`);
    }
    if (events.length > 5) {
      sections.push(`- Versatile event package with ${eventCount} events for various situations`);
    }
    sections.push('');
    sections.push('Potential Weaknesses:');
    if (lowCostCards.length === 0) {
      sections.push('- Limited early game options may struggle against aggressive decks');
    }
    if (highCostCards.length === 0) {
      sections.push('- May lack closing power in longer games');
    }
    sections.push('');
    
    // 4. Key Combos
    sections.push('4. Key Combos and Interactions');
    sections.push('');
    comboHighlights.forEach((combo) => {
      sections.push(`- ${combo}`);
    });
    sections.push('');
    
    // 5. Gameplay Guide
    sections.push('5. Gameplay Guide');
    sections.push('');
    sections.push('Early Game (Turns 1-2):');
    if (lowCostCards.length > 0) {
      const turn1Card = lowCostCards.find((c) => parseInt(c.card.cost ?? '0', 10) === 1);
      const turn2Card = lowCostCards.find((c) => parseInt(c.card.cost ?? '0', 10) === 2);
      if (turn1Card) {
        sections.push(`- Play ${turn1Card.card.name} on turn 1 to establish board presence`);
      }
      if (turn2Card) {
        sections.push(`- Play ${turn2Card.card.name} on turn 2 to continue developing your board`);
      }
      if (isAggro) {
        sections.push('- Attack aggressively to pressure opponent\'s life total');
      } else {
        sections.push('- Focus on setting up your engine and searching for key pieces');
      }
    } else {
      sections.push('- Use leader ability and events to maintain tempo while building resources');
    }
    sections.push('');
    sections.push('Mid Game (Turns 3-4):');
    if (midCostCards.length > 0) {
      const turn3Card = midCostCards.find((c) => parseInt(c.card.cost ?? '0', 10) === 3);
      const turn4Card = midCostCards.find((c) => parseInt(c.card.cost ?? '0', 10) === 4);
      if (turn3Card) {
        sections.push(`- Play ${turn3Card.card.name} to ${isControl ? 'establish board control' : 'maintain pressure'}`);
      }
      if (turn4Card) {
        sections.push(`- Play ${turn4Card.card.name} to ${isControl ? 'stabilize and prepare for late game' : 'push for damage'}`);
      }
    }
    sections.push('- Use events strategically to respond to opponent\'s plays');
    sections.push('- Manage your Don!! resources efficiently');
    sections.push('');
    sections.push('Late Game (Turns 5+):');
    if (highCostCards.length > 0) {
      const finisher = highCostCards[0];
      sections.push(`- Play ${finisher.card.name} as your primary finisher`);
      sections.push(`- Use ${leader.name}'s ability to support your game plan`);
      sections.push('- Close out the game with powerful attacks or game-ending effects');
    } else {
      sections.push(`- Continue applying pressure with your board`);
      sections.push(`- Use ${leader.name}'s ability each turn to maintain advantage`);
      sections.push('- Win through consistent damage and resource management');
    }
    sections.push('');
    
    // 6. Strategies
    sections.push('6. Strategies');
    sections.push('');
    sections.push('Resource Management:');
    sections.push('- Prioritize Don!! cards for key plays - don\'t waste resources on suboptimal turns');
    sections.push('- Maintain hand size by drawing cards when possible');
    sections.push('- Use life as a resource, but be mindful of critical thresholds');
    sections.push('');
    sections.push('Decision Making:');
    if (isAggro) {
      sections.push('- Be aggressive early - pressure opponent\'s life total from the start');
      sections.push('- Prioritize damage over board control in most situations');
    } else if (isControl) {
      sections.push('- Play defensively early - focus on board control and card advantage');
      sections.push('- Save removal for key threats, don\'t waste it on minor targets');
    } else {
      sections.push('- Adapt your playstyle based on the matchup');
      sections.push('- Balance aggression and defense based on board state');
    }
    sections.push('- Prioritize playing cards that advance your game plan');
    sections.push('- Don\'t overextend into board wipes or removal');
    sections.push('');
    sections.push('Matchup Adaptation:');
    sections.push('- Against aggressive decks: Focus on early blockers and removal');
    sections.push('- Against control decks: Apply pressure early and save resources for key threats');
    sections.push('- Against midrange decks: Match their tempo and look for advantageous trades');
    sections.push('');
    
    // 7. Tips and Tricks
    sections.push('7. Tips and Tricks');
    sections.push('');
    sections.push('Common Mistakes to Avoid:');
    sections.push('- Don\'t play too many cards in one turn - maintain hand size');
    sections.push('- Don\'t waste removal on non-threatening targets');
    sections.push('- Don\'t forget to use your leader ability each turn when possible');
    sections.push('');
    sections.push('Optimal Sequencing:');
    sections.push('- Play setup cards before payoff cards');
    sections.push('- Use search effects early to find key pieces');
    sections.push('- Save high-impact events for crucial moments');
    sections.push('');
    sections.push('Maximizing Leader Ability:');
    if (leader.text) {
      sections.push(`- ${leader.name}'s ability: ${leader.text.substring(0, 150)}...`);
      sections.push('- Use this ability proactively each turn when possible');
    } else {
      sections.push(`- Use ${leader.name}'s base stats and color identity to your advantage`);
    }
    sections.push('');
    
    // 8. Competitive Viability
    sections.push('8. Competitive Viability');
    sections.push('');
    sections.push(`This ${strategyType} deck has a solid foundation with ${characterCount} characters and ${eventCount} events.`);
    if (isAggro) {
      sections.push('The aggressive strategy can catch opponents off-guard and close games quickly.');
    } else if (isControl) {
      sections.push('The control strategy provides resilience and late-game power.');
    } else {
      sections.push('The midrange strategy offers flexibility and adaptability.');
    }
    sections.push('Practice with the deck to understand its nuances and optimize your play.');
    sections.push('');
    
    return sections.join('\n');
  }

  private tokenizePrompt(value: string): string[] {
    return value
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, ' ')
      .split(/\s+/)
      .filter((token) => token.length >= 3);
  }

  /**
   * Checks if a card contributes to the deck's purpose and gameplay.
   * Cards must have at least one reason to be in the deck:
   * - Relevance to prompt (keywords, themes)
   * - Synergy with leader abilities
   * - Synergy with other cards in deck
   * - Useful standalone effects
   */
  private async cardContributesToDeck(
    card: import('@shared/types/optcg-card').OptcgCard,
    prompt: string,
    leader: import('@shared/types/optcg-card').OptcgCard,
    existingCards: DeckCardSuggestionWithCard[],
  ): Promise<boolean> {
    const cardText = (card.text ?? '').toLowerCase();
    const cardName = (card.name ?? '').toLowerCase();
    const promptTokens = new Set(this.tokenizePrompt(prompt));
    const leaderName = (leader.name ?? '').toLowerCase();
    const leaderText = (leader.text ?? '').toLowerCase();
    const leaderSubtypes = new Set(
      (leader.subtypes ?? []).map((s) => s.toLowerCase().trim()).filter(Boolean),
    );
    const cardSubtypes = new Set(
      (card.subtypes ?? []).map((s) => s.toLowerCase().trim()).filter(Boolean),
    );

    // 1. Check relevance to prompt (name or text contains prompt keywords)
    const cardTokens = new Set([
      ...this.tokenizePrompt(cardName),
      ...this.tokenizePrompt(cardText),
    ]);
    const promptRelevance = Array.from(promptTokens).some((token) => cardTokens.has(token));

    // 2. Check synergy with leader (card references leader name, subtypes, or abilities)
    const leaderNameToken = leaderName.split(' ')[0];
    let hasLeaderSynergy =
      !!leaderNameToken && (cardName.includes(leaderNameToken) || cardText.includes(leaderNameToken));
    // Check if card shares subtypes with leader (synergy)
    const sharedSubtypes = Array.from(cardSubtypes).filter((s) => leaderSubtypes.has(s));
    if (sharedSubtypes.length > 0) {
      hasLeaderSynergy = true;
    }

    // Check if card text references leader subtypes
    for (const subtype of leaderSubtypes) {
      if (cardText.includes(subtype)) {
        hasLeaderSynergy = true;
        break;
      }
    }

    if (!hasLeaderSynergy && leaderText) {
      const leaderAbilityTokens = this.tokenizePrompt(leaderText).filter((token) => token.length >= 4);
      const abilityMatch = leaderAbilityTokens.some((token) => cardTokens.has(token));
      if (abilityMatch) {
        hasLeaderSynergy = true;
      }
    }

    const supportsLeader = this.supportsLeaderAbility(card, leader);

    // 3. Check synergy with existing cards in deck
    let synergisesWithExisting = false;
    for (const existingCard of existingCards) {
      const existingSubtypes = new Set(
        (existingCard.card.subtypes ?? []).map((s) => s.toLowerCase().trim()).filter(Boolean),
      );
      
      // Shared subtypes indicate synergy
      const sharedWithExisting = Array.from(cardSubtypes).filter((s) => existingSubtypes.has(s));
      if (sharedWithExisting.length > 0) {
        synergisesWithExisting = true;
        break;
      }

      // Check if card text references existing card names or subtypes
      const existingName = (existingCard.card.name ?? '').toLowerCase();
      const existingText = (existingCard.card.text ?? '').toLowerCase();
      
      // Card references existing card by name
      const existingNameToken = existingName.split(' ')[0];
      if (existingNameToken && cardText.includes(existingNameToken)) {
        synergisesWithExisting = true;
        break;
      }

      // Card references existing card's subtypes
      for (const subtype of existingSubtypes) {
        if (cardText.includes(subtype)) {
          synergisesWithExisting = true;
          break;
        }
      }
      if (synergisesWithExisting) {
        break;
      }
    }

    let hasTournamentSignal = false;
    const cardId = card.id ?? '';
    const leaderId = leader.id ?? '';
    if (cardId && leaderId) {
      const existingIds = existingCards
        .map((existing) => existing.card?.id ?? existing.cardSetId ?? '')
        .filter(Boolean);
      hasTournamentSignal = await this.hasTournamentSynergy(cardId, leaderId, existingIds);
    }

    // 6. Check if card is in top tiers (S or A) - these are competitive staples
    let isTopTier = false;
    if (cardId && leaderId) {
      isTopTier = await this.cardTierService.isTopTier(cardId, leaderId);
    }

    return Boolean(
      promptRelevance || hasLeaderSynergy || supportsLeader || synergisesWithExisting || hasTournamentSignal || isTopTier,
    );
  }

  /**
   * Filters out cards that don't contribute to the deck's purpose.
   * Only keeps cards that have at least one reason to be in the deck.
   */
  private async filterRelevantCards(
    cards: DeckCardSuggestionWithCard[],
    prompt: string,
    leader: import('@shared/types/optcg-card').OptcgCard,
  ): Promise<DeckCardSuggestionWithCard[]> {
    const relevant: DeckCardSuggestionWithCard[] = [];
    const existingCards: DeckCardSuggestionWithCard[] = [];

    // Process cards in order of quantity (higher quantity = more important)
    const sorted = cards.slice().sort((a, b) => b.quantity - a.quantity);

    for (const card of sorted) {
      // Check if card contributes to deck
      const contributes = await this.cardContributesToDeck(card.card, prompt, leader, existingCards);

      if (contributes) {
        relevant.push(card);
        existingCards.push(card); // Add to existing for synergy checks
      }
    }

    return relevant;
  }

  /**
   * Intelligently increases quantities of existing cards to reach 50 cards.
   * Prioritizes cards based on tier, role, synergy, and current quantity.
   */
  private async increaseExistingCardQuantities(
    cards: DeckCardSuggestionWithCard[],
    currentTotal: number,
    leaderCardId: string,
    leader: import('@shared/types/optcg-card').OptcgCard | null,
    prompt: string,
    aggressive: boolean = false,
  ): Promise<number> {
    if (currentTotal >= 50 || !leader) {
      return currentTotal;
    }

    // Score each card to determine priority for increasing quantity
    const scoredCards = await Promise.all(
      cards.map(async (card) => {
        let score = 0;
        const cardId = card.card.id ?? '';

        // Base score from current quantity (prefer increasing 2-3 copies to 4)
        // In aggressive mode, also prioritize 1-ofs to reach 50
        if (card.quantity === 1) {
          score += aggressive ? 10 : 5; // Higher priority in aggressive mode
        } else if (card.quantity === 2) {
          score += aggressive ? 20 : 15; // Higher priority in aggressive mode
        } else if (card.quantity === 3) {
          score += 20; // Highest priority - 3-ofs should be 4
        }

        // Tier bonus (S/A tier cards are more important)
        if (cardId) {
          const tier = await this.cardTierService.getCardTier(cardId, leaderCardId);
          if (tier === 'S') {
            score += 25;
          } else if (tier === 'A') {
            score += 15;
          } else if (tier === 'B') {
            score += 5;
          }
        }

        // Role bonus (characters and events are usually more important)
        if (card.role === 'character') {
          score += 10;
        } else if (card.role === 'event') {
          score += 8;
        }

        // Cost bonus (lower cost cards benefit more from multiple copies)
        const cost = parseInt(card.card.cost ?? '0', 10);
        if (cost <= 2) {
          score += 8; // Low cost cards are good to have more copies
        } else if (cost <= 4) {
          score += 5; // Mid cost cards
        } else {
          score += 2; // High cost cards (usually 1-2 copies is enough)
        }

        // Synergy bonus (check if card contributes to deck)
        const contributes = await this.cardContributesToDeck(
          card.card,
          prompt,
          leader,
          cards.filter((c) => c !== card),
        );
        if (contributes) {
          score += 10;
        }

        // Check if it's a key card for the leader
        const leaderStrategy = await this.leaderStrategyService.getStrategyForLeader(leaderCardId);
        const isKeyCard = leaderStrategy?.keyCards.some(
          (kc) => kc.cardId.toUpperCase() === cardId.toUpperCase(),
        );
        if (isKeyCard) {
          score += 20; // Key cards should be maxed out
        }

        return {
          card,
          score,
          room: 4 - card.quantity, // How many more copies we can add
        };
      }),
    );

    // Sort by score (highest first), then by current quantity (lower first for same score)
    scoredCards.sort((a, b) => {
      if (b.score !== a.score) {
        return b.score - a.score;
      }
      return a.card.quantity - b.card.quantity;
    });

    // Increase quantities starting with highest-scored cards
    let newTotal = currentTotal;
    for (const { card, room } of scoredCards) {
      if (newTotal >= 50) {
        break;
      }
      if (room > 0) {
        const add = Math.min(room, 50 - newTotal);
        if (add > 0) {
          card.quantity += add;
          newTotal += add;
        }
      }
    }

    return newTotal;
  }

  /**
   * Adds filler cards that match the leader's type/subtypes as a last resort.
   * Only adds cards that share subtypes with the leader (e.g., Whitebeard Pirates).
   * Modifies existingCards in place by increasing quantities or adding new cards.
   */
  private async addFillerCardsMatchingLeaderType(
    existingCards: DeckCardSuggestionWithCard[],
    currentTotal: number,
    leaderColors: Set<string>,
    leaderCardId: string,
    leaderSubtypes: Set<string>,
    leader: import('@shared/types/optcg-card').OptcgCard | null,
    prompt: string,
    filters: { allowedSetIds: Set<string>; metaOnly: boolean },
  ): Promise<void> {
    if (currentTotal >= 50 || !leader) {
      return;
    }

    let needed = 50 - currentTotal;
    if (needed <= 0) {
      return;
    }

    const existingIds = new Set(existingCards.map((c) => this.getBaseCardCode(c.card.id)));
    existingIds.add(this.getBaseCardCode(leaderCardId));

    // Extract leader family from raw data as fallback
    const leaderFamily = (leader.raw as any)?.family as string | undefined;
    const leaderFamilyParts = leaderFamily
      ? leaderFamily
          .split(/[\/,&]/)
          .map((p) => p.trim().toLowerCase())
          .filter(Boolean)
      : [];
    let allLeaderTypes = new Set([
      ...Array.from(leaderSubtypes).map((s) => s.toLowerCase().trim()),
      ...leaderFamilyParts,
    ]);

    // Special case: if leader has "?" as subtype, extract types from ability text
    if (allLeaderTypes.has('?')) {
      const leaderText = (leader.text ?? '').toLowerCase();
      // Extract types mentioned in ability (e.g., {Celestial Dragons}, {Mary Geoise})
      const abilityTypePattern = /\{([^}]+)\}/g;
      let match: RegExpExecArray | null;
      abilityTypePattern.lastIndex = 0; // Reset regex
      while ((match = abilityTypePattern.exec(leaderText)) !== null) {
        const extractedType = match[1].toLowerCase().trim();
        if (extractedType && extractedType !== '?') {
          allLeaderTypes.add(extractedType);
        }
      }
      // Remove "?" from the set since it's not a real type
      allLeaderTypes.delete('?');
    }

    // If after processing "?" we have no valid types, return early
    if (allLeaderTypes.size === 0) {
      return;
    }

    // Check if all existing cards are already at 4 copies
    // If not, we should not add new filler cards - increaseExistingCardQuantities should handle it
    const allCardsMaxed = existingCards.every((card) => card.quantity >= 4);
    if (!allCardsMaxed) {
      // Not all cards are maxed, so we shouldn't add new filler cards
      // The increaseExistingCardQuantities method should handle increasing them
      return;
    }

    // All existing cards are at 4, so now we can add new filler cards that match leader type
    // First, try to increase quantities of existing cards that match leader type (shouldn't happen if all are 4, but check anyway)
    const matchingExistingCards = existingCards.filter((card) => {
      const cardSubtypes = new Set(
        (card.card.subtypes ?? []).map((s) => s.toLowerCase().trim()).filter(Boolean),
      );
      // Also check raw family field
      const cardFamily = (card.card.raw as any)?.family as string | undefined;
      const cardFamilyParts = cardFamily
        ? cardFamily
            .split(/[\/,&]/)
            .map((p) => p.trim().toLowerCase())
            .filter(Boolean)
        : [];
      const allCardTypes = new Set([...Array.from(cardSubtypes), ...cardFamilyParts]);

      // Check if card shares any type with leader
      return Array.from(allLeaderTypes).some((leaderType) => allCardTypes.has(leaderType));
    });

    // If there are matching cards that aren't at 4, increase them first (shouldn't happen, but safety check)
    const matchingNotMaxed = matchingExistingCards.filter((card) => card.quantity < 4);
    if (matchingNotMaxed.length > 0) {
      // Sort by quantity (lower first to max them out)
      matchingNotMaxed.sort((a, b) => a.quantity - b.quantity);
      
      for (const existing of matchingNotMaxed) {
        if (needed <= 0) break;
        const room = 4 - existing.quantity;
        if (room > 0) {
          const add = Math.min(room, needed);
          if (add > 0) {
            existing.quantity += add;
            const leaderTypeStr = leaderFamily || Array.from(leaderSubtypes).join('/') || 'archetype';
            existing.rationale = `${existing.rationale || ''} Increased to x${existing.quantity} to reach 50 cards. Matches ${leader.name}'s ${leaderTypeStr} type.`.trim();
            needed -= add;
          }
        }
      }
    }

    // Second pass: add new cards that match leader type
    if (needed > 0) {
      const candidateIds = new Set<string>();
      for (const color of leaderColors) {
        const records = await this.cardKnowledge.getCardsByColor(color);
        for (const rec of records) {
          candidateIds.add(rec.id.toUpperCase());
        }
      }

    // Filter to only cards that match leader subtypes
    let filteredCandidateIds = Array.from(candidateIds)
      .filter((id) => {
        const baseCode = this.getBaseCardCode(id);
        return !existingIds.has(baseCode);
      });

    // Shuffle candidate IDs to add randomness to deck generation
    for (let i = filteredCandidateIds.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [filteredCandidateIds[i], filteredCandidateIds[j]] = [filteredCandidateIds[j], filteredCandidateIds[i]];
    }

    filteredCandidateIds = filteredCandidateIds.slice(0, 100); // Limit candidates after shuffling

    if (filteredCandidateIds.length > 0) {
      const cards = await this.optcgApiService.getCardsBySetIds(filteredCandidateIds);
      const leaderCard = leader ? await this.optcgApiService.getCardBySetId(leaderCardId).catch(() => null) : null;

        for (const card of cards) {
          if (needed <= 0) break;
          if (!card) continue;
          if (!this.cardMeetsFilters(card, filters.allowedSetIds, filters.metaOnly, leaderCard ?? undefined)) continue;
          if (card.type?.toLowerCase() === 'leader') continue;
          if (!this.isColorLegal(card.color ?? null, leaderColors)) continue;

          // Must match leader subtypes or family
          const cardSubtypes = new Set(
            (card.subtypes ?? []).map((s) => s.toLowerCase().trim()).filter(Boolean),
          );
          // Also check raw family field
          const cardFamily = (card.raw as any)?.family as string | undefined;
          const cardFamilyParts = cardFamily
            ? cardFamily
                .split(/[\/,&]/)
                .map((p) => p.trim().toLowerCase())
                .filter(Boolean)
            : [];
          const allCardTypes = new Set([...Array.from(cardSubtypes), ...cardFamilyParts]);

          const matchesLeaderType = Array.from(allLeaderTypes).some((leaderType) =>
            allCardTypes.has(leaderType),
          );

          if (!matchesLeaderType) {
            continue;
          }

          // Check for conflicts
          if (this.hasConflictsWithLeader(card, leader, leaderColors, existingCards)) {
            continue;
          }

          // Exclude vanilla cards
          if (this.isVanillaCard(card)) {
            continue;
          }

          const baseCode = this.getBaseCardCode(card.id);
          if (existingIds.has(baseCode)) {
            continue;
          }

          const qty = Math.min(2, needed); // Start with 2 copies for filler
          if (qty > 0) {
            const leaderTypeStr = leaderFamily || Array.from(leaderSubtypes).join('/') || 'archetype';
            existingCards.push({
              cardSetId: card.id,
              quantity: qty,
              rationale: `x${qty}: Filler card added to reach 50 cards. Matches ${leader.name}'s ${leaderTypeStr} type.`,
              role: this.resolveRole(card),
              card,
            });
            existingIds.add(baseCode);
            needed -= qty;
          }
        }
      }
    }
  }

  /**
   * Enhances card rationales with tier information and leader strategy context.
   */
  private async enhanceCardRationales(
    cards: DeckCardSuggestionWithCard[],
    leaderCardId: string,
    leader: import('@shared/types/optcg-card').OptcgCard | null,
  ): Promise<DeckCardSuggestionWithCard[]> {
    if (!leader) {
      return cards;
    }

    // Get leader strategy for context
    const leaderStrategy = await this.leaderStrategyService.getStrategyForLeader(leaderCardId);
    const keyCards = leaderStrategy?.keyCards || [];

    return Promise.all(
      cards.map(async (card) => {
        const cardId = card.card.id ?? '';
        if (!cardId) {
          return card;
        }

        const enhancements: string[] = [];

        // Add tier information
        const tier = await this.cardTierService.getCardTier(cardId, leaderCardId);
        if (tier) {
          const tierDescriptions: Record<string, string> = {
            S: 'Top-tier competitive staple',
            A: 'High-tier competitive card',
            B: 'Good role player',
            C: 'Situational card',
          };
          enhancements.push(`Tier ${tier}: ${tierDescriptions[tier]}.`);
        }

        // Check if card is a key card for this leader
        const isKeyCard = keyCards.some((kc) => kc.cardId.toUpperCase() === cardId.toUpperCase());
        if (isKeyCard) {
          const keyCardInfo = keyCards.find((kc) => kc.cardId.toUpperCase() === cardId.toUpperCase());
          if (keyCardInfo) {
            enhancements.push(`Key card for ${leader.name}: ${keyCardInfo.rationale}`);
          }
        }

        // Enhance existing rationale
        let enhancedRationale = card.rationale || '';
        if (enhancements.length > 0) {
          const enhancementText = enhancements.join(' ');
          if (enhancedRationale) {
            enhancedRationale = `${enhancementText} ${enhancedRationale}`;
          } else {
            enhancedRationale = enhancementText;
          }
        }

        return {
          ...card,
          rationale: enhancedRationale,
        };
      }),
    );
  }

  private enforceThemeDistribution(
    prompt: string,
    leader: import('@shared/types/optcg-card').OptcgCard,
    cards: DeckCardSuggestionWithCard[],
    target: number = 50,
  ): DeckCardSuggestionWithCard[] {
    const promptTokens = new Set([
      ...this.tokenizePrompt(prompt),
      ...this.tokenizePrompt(leader.name ?? ''),
    ]);

    const scored = cards.map((card, index) => {
      const nameTokens = this.tokenizePrompt(card.card.name ?? '');
      const matchesPrompt = nameTokens.some((token) => promptTokens.has(token));
      let score = 0;
      if (matchesPrompt) score += 8;
      if (card.role === 'event') score += 2;
      if (card.role === 'character') score += 1;
      score += card.quantity * 0.5;

      // Prefer higher quantities for consistency
      // Important cards (prompt-matching) should have 3-4 copies
      // Regular cards should have 2-3 copies minimum
      // Only niche cards should have 1 copy
      let desired = card.quantity;
      if (matchesPrompt) {
        // Prompt-matching cards are important - prefer 4 copies
        desired = Math.max(desired, 4);
      } else if (card.role === 'event') {
        // Events are usually important - prefer 2-3 copies
        desired = Math.max(desired, 2);
      } else if (card.role === 'character') {
        // Characters are core - prefer 2-3 copies
        desired = Math.max(desired, 2);
      } else {
        // Other cards (stages, etc.) can be 1-2 copies
        desired = Math.max(desired, 1);
      }
      desired = Math.min(4, desired);

      return { card: { ...card }, score, desired, assigned: 0, index, matchesPrompt };
    });

    scored.sort((a, b) => b.score - a.score);

    let total = 0;
    for (const entry of scored) {
      // Ensure minimum quantities: prompt-matching cards get 3+, core cards get 2+
      let minQty = 1;
      if (entry.matchesPrompt) {
        minQty = 3; // Prompt-matching cards should have at least 3 copies
      } else if (entry.card.role === 'event' || entry.card.role === 'character') {
        minQty = 2; // Core cards should have at least 2 copies
      }
      
      entry.assigned = Math.max(minQty, Math.min(4, entry.desired));
      total += entry.assigned;
    }

    if (total > target) {
      let excess = total - target;
      // Sort by score (lowest first) and quantity (highest first) to reduce from least important cards
      const donors = scored.slice().sort((a, b) => a.score - b.score || b.assigned - a.assigned);
      
      // First pass: reduce from cards that can go below their minimum
      // Try to keep prompt-matching cards at 3+, regular cards at 2+
      for (const entry of donors) {
        if (excess <= 0) break;
        // Set minimum quantities based on importance
        let minQty = 1;
        if (entry.matchesPrompt) {
          minQty = 3; // Keep prompt-matching cards at 3+ copies
        } else if (entry.card.role === 'event' || entry.card.role === 'character') {
          minQty = 2; // Keep core cards at 2+ copies
        }
        
        const reducible = Math.max(entry.assigned - minQty, 0);
        const reduceBy = Math.min(reducible, excess);
        if (reduceBy > 0) {
          entry.assigned -= reduceBy;
          excess -= reduceBy;
        }
      }
      
      // Second pass: if still excess, reduce further but try to keep at least 2 for important cards
      if (excess > 0) {
        for (const entry of donors) {
          if (excess <= 0) break;
          let minQty = 1;
          if (entry.matchesPrompt) {
            minQty = 2; // Don't go below 2 for prompt-matching cards
          } else if (entry.card.role === 'event' || entry.card.role === 'character') {
            minQty = 1; // Can go to 1 for regular cards if needed
          }
          
          const reducible = Math.max(entry.assigned - minQty, 0);
          const reduceBy = Math.min(reducible, excess);
          if (reduceBy > 0) {
            entry.assigned -= reduceBy;
            excess -= reduceBy;
          }
        }
      }
      
      // Final pass: if still excess, reduce everything to minimum 1
      if (excess > 0) {
        for (const entry of donors) {
          if (excess <= 0) break;
          const reducible = Math.max(entry.assigned - 1, 0);
          const reduceBy = Math.min(reducible, excess);
          if (reduceBy > 0) {
            entry.assigned -= reduceBy;
            excess -= reduceBy;
          }
        }
      }
    }

    let currentTotal = scored.reduce((sum, entry) => sum + entry.assigned, 0);
    if (currentTotal < target) {
      let remaining = target - currentTotal;
      // Prioritize increasing quantities of important cards first
      const receivers = scored.slice().sort((a, b) => {
        // First sort by score (highest first)
        if (b.score !== a.score) return b.score - a.score;
        // Then by current quantity (lowest first) to boost cards that need more copies
        return a.assigned - b.assigned;
      });
      
      // First pass: boost important cards to their preferred quantities
      for (const entry of receivers) {
        if (remaining <= 0) break;
        let preferredQty = 4;
        if (entry.matchesPrompt) {
          preferredQty = 4; // Prompt-matching cards should be 4
        } else if (entry.card.role === 'event' || entry.card.role === 'character') {
          preferredQty = 3; // Core cards should be 2-3
        } else {
          preferredQty = 2; // Other cards can be 1-2
        }
        
        const room = Math.min(4 - entry.assigned, preferredQty - entry.assigned);
        if (room > 0) {
          const add = Math.min(room, remaining);
          entry.assigned += add;
          remaining -= add;
        }
      }
      
      // Second pass: distribute remaining cards to fill to 50
      if (remaining > 0) {
        for (const entry of receivers) {
          if (remaining <= 0) break;
          const room = 4 - entry.assigned;
          if (room > 0) {
            const add = Math.min(room, remaining);
            entry.assigned += add;
            remaining -= add;
          }
        }
      }
      
      currentTotal = scored.reduce((sum, entry) => sum + entry.assigned, 0);
    }

    const finalCards = scored
      .filter((entry) => entry.assigned > 0)
      .map((entry) => ({
        ...entry.card,
        quantity: entry.assigned,
      }))
      .sort((a, b) => {
        const tierA = entryTier(a.quantity);
        const tierB = entryTier(b.quantity);
        return tierA === tierB
          ? a.cardSetId.localeCompare(b.cardSetId)
          : tierA - tierB;
      });

    return finalCards;

    function entryTier(quantity: number): number {
      if (quantity >= 4) return 1;
      if (quantity >= 3) return 2;
      if (quantity >= 2) return 3;
      return 4;
    }
  }

  /**
   * Consolidates singleton (x1) cards to improve deck consistency.
   * Removes less important x1 cards and boosts important ones to 2+ copies.
   */
  private consolidateSingletonCards(
    cards: DeckCardSuggestionWithCard[],
    prompt: string,
    leader: import('@shared/types/optcg-card').OptcgCard,
  ): DeckCardSuggestionWithCard[] {
    const result = cards.map((c) => ({ ...c }));
    const total = result.reduce((sum, c) => sum + c.quantity, 0);
    
    // Count singleton cards
    const singletons = result.filter((c) => c.quantity === 1);
    const nonSingletons = result.filter((c) => c.quantity > 1);
    
    // If we have too many singletons (8 or more), consolidate them
    // Tournament decks typically have 0-3 singletons, so 8+ is excessive
    if (singletons.length >= 8) {
      const promptTokens = new Set(this.tokenizePrompt(prompt));
      const leaderName = (leader.name ?? '').toLowerCase();
      
      // Score singletons by importance
      const scoredSingletons = singletons.map((card) => {
        const cardName = (card.card.name ?? '').toLowerCase();
        const cardText = (card.card.text ?? '').toLowerCase();
        const nameTokens = this.tokenizePrompt(cardName);
        const matchesPrompt = nameTokens.some((token) => promptTokens.has(token));
        const matchesLeader = cardName.includes(leaderName.split(' ')[0]) || cardText.includes(leaderName.split(' ')[0]);
        
        let score = 0;
        if (matchesPrompt) score += 10;
        if (matchesLeader) score += 5;
        if (card.role === 'character') score += 3;
        if (card.role === 'event') score += 2;
        
        // Check for useful effects
        if (cardText.includes('draw') || cardText.includes('search') || cardText.includes('k.o.')) {
          score += 2;
        }
        
        return { card, score };
      });
      
      // Sort by score (lowest first - these are candidates for removal)
      scoredSingletons.sort((a, b) => a.score - b.score);
      
      // Remove the least important singletons (keep top 3-5 most important)
      // For 8 singletons: keep 4, remove 4
      // For 10+ singletons: keep 5, remove the rest
      const keepCount = singletons.length >= 10 ? 5 : Math.max(3, Math.floor(singletons.length / 2));
      const toRemove = scoredSingletons.slice(0, Math.max(0, singletons.length - keepCount));
      const toKeep = scoredSingletons.slice(Math.max(0, singletons.length - keepCount));
      
      // Remove low-scoring singletons
      const removedIds = new Set(toRemove.map((s) => this.getBaseCardCode(s.card.card.id)));
      const filtered = result.filter((c) => !removedIds.has(this.getBaseCardCode(c.card.id)));
      
      // Boost important singletons to 2 copies
      const remaining = 50 - filtered.reduce((sum, c) => sum + c.quantity, 0);
      let available = remaining;
      
      // Sort kept singletons by score (highest first)
      toKeep.sort((a, b) => b.score - a.score);
      
      for (const { card } of toKeep) {
        if (available <= 0) break;
        const existing = filtered.find((c) => this.getBaseCardCode(c.card.id) === this.getBaseCardCode(card.card.id));
        if (existing && existing.quantity === 1) {
          const boost = Math.min(1, available); // Boost from 1 to 2
          existing.quantity += boost;
          available -= boost;
        }
      }
      
      // Distribute remaining cards to non-singletons
      if (available > 0) {
        const sorted = filtered
          .filter((c) => c.quantity < 4)
          .sort((a, b) => {
            // Prioritize cards with 2-3 copies (they're important but not maxed)
            if (a.quantity === 2 || a.quantity === 3) return -1;
            if (b.quantity === 2 || b.quantity === 3) return 1;
            return b.quantity - a.quantity;
          });
        
        for (const card of sorted) {
          if (available <= 0) break;
          const room = 4 - card.quantity;
          if (room > 0) {
            const add = Math.min(room, available);
            card.quantity += add;
            available -= add;
          }
        }
      }
      
      return filtered;
    }
    
    // If we have fewer singletons (1-7), just try to boost important ones
    if (singletons.length > 0 && singletons.length < 8) {
      const promptTokens = new Set(this.tokenizePrompt(prompt));
      const leaderName = (leader.name ?? '').toLowerCase();
      
      // Score and boost important singletons
      const scored = singletons.map((card) => {
        const cardName = (card.card.name ?? '').toLowerCase();
        const nameTokens = this.tokenizePrompt(cardName);
        const matchesPrompt = nameTokens.some((token) => promptTokens.has(token));
        const matchesLeader = cardName.includes(leaderName.split(' ')[0]);
        
        let score = 0;
        if (matchesPrompt) score += 10;
        if (matchesLeader) score += 5;
        if (card.role === 'character' || card.role === 'event') score += 3;
        
        return { card, score };
      });
      
      scored.sort((a, b) => b.score - a.score);
      
      // Boost top singletons to 2 copies if we have room
      let available = 50 - total;
      for (const { card } of scored) {
        if (available <= 0) break;
        const existing = result.find((c) => this.getBaseCardCode(c.card.id) === this.getBaseCardCode(card.card.id));
        if (existing && existing.quantity === 1) {
          const boost = Math.min(1, available);
          existing.quantity += boost;
          available -= boost;
        }
      }
    }
    
    return result;
  }

  private metadataMeetsFilters(cardId: string, allowedSetIds: Set<string>, metaOnly: boolean): boolean {
    if (this.cardKnowledge.isBannedCard(cardId)) {
      return false;
    }
    if (metaOnly && !this.cardKnowledge.isMetaCard(cardId)) {
      return false;
    }
    if (allowedSetIds.size > 0) {
      const setCode = this.getSetCode(cardId);
      const normalizedSetCode = this.normalizeSetCode(setCode);
      if (!allowedSetIds.has(normalizedSetCode)) {
        return false;
      }
    }
    return true;
  }
}

