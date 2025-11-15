import { Injectable, Logger } from '@nestjs/common';
import { DeckCardSuggestion } from '@shared/contracts/ai';
import { LeaderKnowledgeService, LeaderKnowledgeMetadata } from './leader-knowledge.service';
import {
  CardKnowledgeMetadata,
  CardKnowledgeService,
} from './card-knowledge.service';
import { PromptIntentAnalysis } from './intent-matcher.service';

export type LocalDeckSuggestion = {
  leaderId: string;
  leaderName: string;
  summary: string;
  cards: DeckCardSuggestion[];
  totalCards: number;
  notes: string[];
  usedKeywords: string[];
};

type ScoredCard = {
  metadata: CardKnowledgeMetadata;
  score: number;
  matchedKeywords: string[];
  matchedColors: string[];
  matchedSubtypes: string[];
  matchedAttributes: string[];
  isExplicit: boolean;
};

type WeightedCard = ScoredCard & {
  quantity: number;
  rationale: string;
  role: DeckCardSuggestion['role'];
};

@Injectable()
export class LocalDeckBuilderService {
  private readonly logger = new Logger(LocalDeckBuilderService.name);
  private readonly maxDeckSize = 50;
  private readonly maxCopiesPerCard = 4;

  constructor(
    private readonly leaderKnowledge: LeaderKnowledgeService,
    private readonly cardKnowledge: CardKnowledgeService,
  ) {}

  async buildDeck(params: {
    leaderId: string;
    intent: PromptIntentAnalysis;
  }): Promise<LocalDeckSuggestion | null> {
    const { leaderId, intent } = params;
    const leaderMetadata = this.leaderKnowledge.getLeaderMetadata(leaderId);
    if (!leaderMetadata) {
      this.logger.warn(`Unable to build local deck: leader ${leaderId} not found in knowledge base`);
      return null;
    }

    const intentKeywordSet = new Set<string>([...intent.keywords, ...intent.tokens]);
    const intentCardIds = new Set<string>(intent.cardIds.map((id) => id.toUpperCase()));
    const targetColors = new Set<string>([...leaderMetadata.colors, ...intent.colors.map((color) => color.toLowerCase())]);
    if (intent.colorHints.length) {
      intent.colorHints.forEach((hint) => targetColors.add(hint.color.toLowerCase()));
    }

    const leaderSubtypeSet = new Set<string>(leaderMetadata.subtypes.map((subtype) => this.normalize(subtype)));

    const cards = await this.cardKnowledge.listAllMetadata();
    const scored = cards
      .filter((card) => !card.isVanilla)
      .filter((card) => card.type.toLowerCase() !== 'leader')
      .map((card) => this.scoreCard(card, {
        intentKeywordSet,
        targetColors,
        leaderSubtypeSet,
        intentCardIds,
        leaderMetadata,
      }))
      .filter((scoredCard): scoredCard is ScoredCard => scoredCard.score > 0);

    if (!scored.length) {
      this.logger.warn(`No heuristic matches found for leader ${leaderId}`);
      return null;
    }

    const weighted = this.selectAndWeightCards(scored);
    const balanced = this.balanceDeck(weighted, scored);

    const deckCards: DeckCardSuggestion[] = balanced.map((card) => ({
      cardSetId: card.metadata.id,
      quantity: card.quantity,
      role: card.role,
      rationale: card.rationale,
    }));

    const totalCards = deckCards.reduce((sum, entry) => sum + entry.quantity, 0);
    const usedKeywords = new Set<string>();
    balanced.forEach((card) => card.matchedKeywords.forEach((keyword) => usedKeywords.add(keyword)));

    const notes = this.generateSummaryNotes(leaderMetadata, deckCards, balanced);

    const summary = this.buildSummary(leaderMetadata, targetColors, usedKeywords);

    return {
      leaderId,
      leaderName: leaderMetadata.name,
      summary,
      cards: deckCards,
      totalCards,
      notes,
      usedKeywords: Array.from(usedKeywords),
    };
  }

  private scoreCard(
    card: CardKnowledgeMetadata,
    context: {
      intentKeywordSet: Set<string>;
      targetColors: Set<string>;
      leaderSubtypeSet: Set<string>;
      intentCardIds: Set<string>;
      leaderMetadata: LeaderKnowledgeMetadata;
    },
  ): ScoredCard {
    const role = this.mapRole(card.type);
    if (role === 'don') {
      return {
        metadata: card,
        score: 0,
        matchedKeywords: [],
        matchedColors: [],
        matchedSubtypes: [],
        matchedAttributes: [],
        isExplicit: false,
      };
    }

    const matchedColors = card.colors.filter((color) => context.targetColors.has(color));
    const matchedKeywords = card.keywords.filter((keyword) => context.intentKeywordSet.has(keyword));
    const matchedSubtypes = card.subtypes
      .map((subtype) => this.normalize(subtype))
      .filter((subtype) => context.leaderSubtypeSet.has(subtype) || context.intentKeywordSet.has(subtype));

    const matchedAttributes: string[] = [];
    if (card.attribute) {
      const normalizedAttribute = this.normalize(card.attribute);
      if (context.intentKeywordSet.has(normalizedAttribute)) {
        matchedAttributes.push(normalizedAttribute);
      }
    }

    if (card.setName) {
      const normalizedSet = this.normalize(card.setName);
      if (context.intentKeywordSet.has(normalizedSet)) {
        matchedKeywords.push(normalizedSet);
      }
    }

    const isExplicit = context.intentCardIds.has(card.id.toUpperCase());

    let score = 0;
    score += matchedColors.length * 4;
    score += matchedKeywords.length * 2;
    score += matchedSubtypes.length * 3;
    score += matchedAttributes.length * 2;

    if (isExplicit) {
      score += 30;
    }

    if (!matchedColors.length && role === 'character') {
      // characters without color alignment are less valuable
      score -= 2;
    }

    if (card.counter > 0) {
      score += 1;
    }

    if (card.text && context.leaderMetadata.name && card.text.toLowerCase().includes(context.leaderMetadata.name.toLowerCase())) {
      score += 6;
      matchedKeywords.push(this.normalize(context.leaderMetadata.name));
    }

    return {
      metadata: card,
      score,
      matchedKeywords: Array.from(new Set(matchedKeywords)),
      matchedColors,
      matchedSubtypes,
      matchedAttributes,
      isExplicit,
    };
  }

  private selectAndWeightCards(scoredCards: ScoredCard[]): WeightedCard[] {
    const sorted = scoredCards.sort((a, b) => {
      if (a.isExplicit !== b.isExplicit) {
        return a.isExplicit ? -1 : 1;
      }
      return b.score - a.score;
    });

    const selections: WeightedCard[] = [];
    let totalCopies = 0;

    for (const scored of sorted) {
      if (totalCopies >= this.maxDeckSize && selections.length >= 20) {
        break;
      }

      const quantity = this.determineQuantity(scored.score, scored.isExplicit);
      if (quantity <= 0) {
        continue;
      }

      const role = this.mapRole(scored.metadata.type);
      const rationale = this.buildCardRationale(scored);

      selections.push({
        ...scored,
        quantity,
        rationale,
        role,
      });

      totalCopies += quantity;
    }

    return selections;
  }

  private balanceDeck(selected: WeightedCard[], allScored: ScoredCard[]): WeightedCard[] {
    if (!selected.length) {
      return selected;
    }

    const target = this.maxDeckSize;
    let total = selected.reduce((sum, card) => sum + card.quantity, 0);

    const remaining = allScored
      .filter((card) => !selected.some((selectedCard) => selectedCard.metadata.id === card.metadata.id))
      .sort((a, b) => b.score - a.score);

    // Attempt to fill up to target using remaining cards
    for (const card of remaining) {
      if (total >= target) {
        break;
      }
      const role = this.mapRole(card.metadata.type);
      if (role === 'don') {
        continue;
      }
      const fillQuantity = Math.min(this.determineQuantity(card.score, card.isExplicit), target - total, this.maxCopiesPerCard);
      if (fillQuantity <= 0) {
        continue;
      }
      selected.push({
        ...card,
        quantity: fillQuantity,
        rationale: this.buildCardRationale(card),
        role,
      });
      total += fillQuantity;
    }

    // Adjust quantities to hit target exactly
    if (total < target) {
      const byScoreDesc = selected.slice().sort((a, b) => b.score - a.score);
      let index = 0;
      while (total < target && index < byScoreDesc.length) {
        const card = byScoreDesc[index];
        const available = Math.min(this.maxCopiesPerCard - card.quantity, target - total);
        if (available > 0) {
          card.quantity += available;
          total += available;
        } else {
          index += 1;
        }
      }
    } else if (total > target) {
      const byScoreAsc = selected.slice().sort((a, b) => a.score - b.score);
      let index = 0;
      while (total > target && index < byScoreAsc.length) {
        const card = byScoreAsc[index];
        const reducible = Math.min(card.quantity - 1, total - target);
        if (reducible > 0) {
          card.quantity -= reducible;
          total -= reducible;
        }
        if (card.quantity <= 1) {
          index += 1;
        }
      }
    }

    // Ensure no card has zero quantity and limit to max copies
    return selected
      .filter((card) => card.quantity > 0)
      .map((card) => ({
        ...card,
        quantity: Math.min(card.quantity, this.maxCopiesPerCard),
      }));
  }

  private determineQuantity(score: number, isExplicit: boolean): number {
    if (isExplicit) {
      return Math.min(this.maxCopiesPerCard, 4);
    }

    if (score >= 20) {
      return 4;
    }
    if (score >= 12) {
      return 3;
    }
    if (score >= 7) {
      return 2;
    }
    if (score >= 4) {
      return 2;
    }
    return 1;
  }

  private buildCardRationale(card: ScoredCard): string {
    const parts: string[] = [];
    if (card.matchedColors.length) {
      parts.push(`Matches deck colors (${card.matchedColors.join(', ')})`);
    }
    if (card.matchedSubtypes.length) {
      parts.push(`Supports ${card.matchedSubtypes.join(', ')}`);
    }
    if (card.matchedKeywords.length) {
      parts.push(`Hits prompt themes: ${card.matchedKeywords.join(', ')}`);
    }
    if (card.matchedAttributes.length) {
      parts.push(`Aligns with attributes: ${card.matchedAttributes.join(', ')}`);
    }
    if (card.metadata.counter > 0) {
      parts.push(`Provides ${card.metadata.counter} counter value`);
    }
    return parts.length ? parts.join('. ') : 'Synergizes with the selected leader strategy.';
  }

  private generateSummaryNotes(
    leader: LeaderKnowledgeMetadata,
    deckCards: DeckCardSuggestion[],
    weightedCards: WeightedCard[],
  ): string[] {
    const totalByRole = new Map<DeckCardSuggestion['role'], number>();
    deckCards.forEach((card) => {
      totalByRole.set(card.role, (totalByRole.get(card.role) ?? 0) + card.quantity);
    });

    const primaryRole = Array.from(totalByRole.entries()).sort((a, b) => b[1] - a[1])[0]?.[0];

    const topCards = weightedCards
      .slice(0, 5)
      .map((card) => `${card.metadata.name} (${card.quantity}x)`);

    const notes: string[] = [];
    if (primaryRole) {
      notes.push(`Primary composition leans on ${primaryRole} cards (${totalByRole.get(primaryRole)} copies).`);
    }
    notes.push(`Top picks: ${topCards.join(', ')}.`);
    notes.push(`Leader subtypes emphasized: ${leader.subtypes.join(', ')}.`);

    return notes;
  }

  private buildSummary(
    leader: LeaderKnowledgeMetadata,
    colors: Set<string>,
    keywords: Set<string>,
  ): string {
    const colorList = Array.from(colors).map((color) => color.charAt(0).toUpperCase() + color.slice(1));
    const keywordList = Array.from(keywords).slice(0, 5).map((keyword) => keyword.replace(/\b\w/g, (char) => char.toUpperCase()));
    return `${leader.name} ${colorList.join('/')} build focusing on ${keywordList.join(', ')}`;
  }

  private mapRole(type: string): DeckCardSuggestion['role'] {
    const normalized = type.trim().toLowerCase();
    if (normalized === 'character') {
      return 'character';
    }
    if (normalized === 'event') {
      return 'event';
    }
    if (normalized === 'stage') {
      return 'stage';
    }
    if (normalized.includes('don')) {
      return 'don';
    }
    return 'other';
  }

  private normalize(value: string): string {
    return value.toLowerCase().replace(/[^a-z0-9\s]/g, ' ').replace(/\s+/g, ' ').trim();
  }
}
