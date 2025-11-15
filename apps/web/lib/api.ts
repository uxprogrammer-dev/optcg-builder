import axios from 'axios';
import {
  CardSuggestionResponse,
  GeneratedDeck,
  LeaderSuggestionWithCard,
  DeckCharacteristicId,
  ReviewDeckRequest,
  ReviewDeckResponse,
} from '@shared/contracts/ai';

const apiClient = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_BASE_URL ?? '/api',
  headers: {
    'ngrok-skip-browser-warning': 'true',
  },
});

export interface ProgressUpdate {
  id: string;
  phase: 'leaders' | 'deck' | null;
  step: string;
  progress: number;
  details?: string;
  timestamp: number;
}

export interface LeaderSuggestionsResponse {
  prompt: string;
  leaders: LeaderSuggestionWithCard[];
  progressId?: string;
}

export interface SetOption {
  id: string;
  label: string;
}

export interface DeckGenerationResponse extends GeneratedDeck {
  totalCards: number;
  progressId?: string;
}

export interface CardSuggestionPayload {
  leaderCardId: string;
  cardIdQuery?: string;
  characteristicId?: DeckCharacteristicId;
  excludeCardIds?: string[];
  limit?: number;
}

export interface LeaderFilters {
  setIds?: string[];
  metaOnly?: boolean;
  tournamentOnly?: boolean;
  regenerate?: boolean;
  progressId?: string;
}

export interface DeckGenerationOptions {
  useOpenAi?: boolean;
  setIds?: string[];
  metaOnly?: boolean;
  tournamentOnly?: boolean;
  progressId?: string;
}

export async function fetchLeaderSuggestions(
  prompt: string,
  options?: LeaderFilters & { useOpenAi?: boolean },
): Promise<LeaderSuggestionsResponse> {
  const { data } = await apiClient.post<LeaderSuggestionsResponse>('/ai/leaders', {
    prompt,
    useOpenAi: options?.useOpenAi,
    setIds: options?.setIds,
    metaOnly: options?.metaOnly,
    tournamentOnly: options?.tournamentOnly,
    regenerate: options?.regenerate,
    progressId: options?.progressId,
  });
  return data;
}

export async function generateDeck(
  prompt: string,
  leaderCardId: string,
  options?: DeckGenerationOptions,
): Promise<DeckGenerationResponse> {
  const { data } = await apiClient.post<DeckGenerationResponse>('/decks', {
    prompt,
    leaderCardId,
    useOpenAi: options?.useOpenAi,
    setIds: options?.setIds,
    metaOnly: options?.metaOnly,
    tournamentOnly: options?.tournamentOnly,
    progressId: options?.progressId,
  });
  return data;
}

export async function downloadDeckArchive(deck: GeneratedDeck): Promise<Blob> {
  const { data } = await apiClient.post<ArrayBuffer>('/decks/download', deck, {
    responseType: 'arraybuffer',
  });

  return new Blob([data], { type: 'application/zip' });
}

export async function suggestDeckCards(payload: CardSuggestionPayload): Promise<CardSuggestionResponse> {
  const { data } = await apiClient.post<CardSuggestionResponse>('/decks/cards/suggest', payload);
  return data;
}

export async function rerunDeckReview(payload: ReviewDeckRequest): Promise<ReviewDeckResponse> {
  const { data } = await apiClient.post<ReviewDeckResponse>('/decks/review', payload);
  return data;
}

export async function fetchSetOptions(): Promise<SetOption[]> {
  const { data } = await apiClient.get<SetOption[]>('/ai/sets');
  return data;
}

export async function fetchProgress(progressId: string): Promise<ProgressUpdate> {
  const { data } = await apiClient.get<ProgressUpdate>(`/ai/progress/${progressId}`);
  return data;
}
