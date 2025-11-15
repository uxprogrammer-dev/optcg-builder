export interface PromptLeaderRequest {
  prompt: string;
  setIds?: string[];
  metaOnly?: boolean;
  tournamentOnly?: boolean;
  regenerate?: boolean;
}

export type LeaderSuggestionsRequest = PromptLeaderRequest;

export interface LeaderSuggestion {
  cardSetId: string;
  leaderName?: string;
  rationale: string;
  strategySummary?: string;
}

export interface LeaderSuggestionWithCard extends LeaderSuggestion {
  card: import('../types/optcg-card').OptcgCard;
}

export interface GenerateDeckRequest {
  prompt: string;
  leaderCardId: string;
  setIds?: string[];
  metaOnly?: boolean;
  tournamentOnly?: boolean;
}

export interface DeckCardSuggestion {
  cardSetId: string;
  quantity: number;
  role: 'character' | 'event' | 'stage' | 'don' | 'counter' | 'other';
  rationale?: string;
}

export interface GeneratedDeck {
  deckId: string;
  summary: string;
  leader: import('../types/optcg-card').OptcgCard;
  cards: DeckCardSuggestionWithCard[];
  source?: 'local' | 'ml' | 'openai';
  notes?: string[];
  gameplaySummary?: string;
  comboHighlights?: string[];
  deckReview?: string;
}

export interface DeckCardSuggestionWithCard extends DeckCardSuggestion {
  card: import('../types/optcg-card').OptcgCard;
}

export type DeckCharacteristicId =
  | 'control'
  | 'cardAdvantage'
  | 'removal'
  | 'trashManipulation'
  | 'aggression'
  | 'defense'
  | 'synergy'
  | 'economy';

export interface CardSuggestionRequest {
  leaderCardId: string;
  cardIdQuery?: string;
  characteristicId?: DeckCharacteristicId;
  excludeCardIds?: string[];
  limit?: number;
}

export interface CardSuggestion {
  card: import('../types/optcg-card').OptcgCard;
  normalizedScore: number;
  rawScore: number;
  characteristicId?: DeckCharacteristicId;
  rationale?: string;
}

export interface CardSuggestionResponse {
  characteristicId?: DeckCharacteristicId;
  cards: CardSuggestion[];
}

export interface ReviewDeckCard {
  cardSetId: string;
  quantity: number;
}

export interface ReviewDeckRequest {
  prompt: string;
  leaderCardId: string;
  cards: ReviewDeckCard[];
}

export interface ReviewDeckResponse {
  deckReview: string;
}

