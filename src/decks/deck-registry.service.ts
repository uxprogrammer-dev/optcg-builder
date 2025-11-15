import { Injectable } from '@nestjs/common';
import { DeckCardSuggestionWithCard, GeneratedDeck } from '@shared/contracts/ai';
import { OptcgCard } from '@shared/types/optcg-card';
import { v4 as uuidv4 } from 'uuid';

interface DeckRecord extends GeneratedDeck {
  prompt: string;
  leaderCardId: string;
  createdAt: number;
  filters?: {
    setIds: string[];
    metaOnly: boolean;
  };
}

@Injectable()
export class DeckRegistryService {
  private readonly deckTtlMs = 1000 * 60 * 60; // 1 hour
  private readonly decks = new Map<string, DeckRecord>();

  createDeck({
    prompt,
    summary,
    leader,
    leaderCardId,
    cards,
    gameplaySummary,
    comboHighlights,
    source,
    notes,
    deckReview,
    filters,
  }: {
    prompt: string;
    summary: string;
    leader: OptcgCard;
    leaderCardId: string;
    cards: DeckCardSuggestionWithCard[];
    gameplaySummary?: string;
    comboHighlights?: string[];
    source?: 'local' | 'ml' | 'openai';
    notes?: string[];
    deckReview?: string;
    filters?: {
      setIds: string[];
      metaOnly: boolean;
    };
  }): DeckRecord {
    this.pruneExpiredDecks();

    const deckId = uuidv4();
    const record: DeckRecord = {
      deckId,
      prompt,
      summary,
      leader,
      leaderCardId,
      cards,
      gameplaySummary,
      comboHighlights,
      source,
      notes,
      deckReview,
      createdAt: Date.now(),
      filters,
    };

    this.decks.set(deckId, record);
    return record;
  }

  getDeck(deckId: string): DeckRecord | null {
    const deck = this.decks.get(deckId);
    if (!deck) {
      return null;
    }

    if (Date.now() - deck.createdAt > this.deckTtlMs) {
      this.decks.delete(deckId);
      return null;
    }

    return deck;
  }

  private pruneExpiredDecks(): void {
    const now = Date.now();
    for (const [deckId, record] of this.decks.entries()) {
      if (now - record.createdAt > this.deckTtlMs) {
        this.decks.delete(deckId);
      }
    }
  }
}

