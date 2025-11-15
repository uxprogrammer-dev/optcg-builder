import { useMemo, useState } from 'react';
import Image from 'next/image';
import {
  CardSuggestion,
  DeckCardSuggestionWithCard,
  DeckCharacteristicId,
} from '@shared/contracts/ai';
import { OptcgCard } from '@shared/types/optcg-card';
import { Dialog } from './dialog';
import { CardSuggestionPayload, suggestDeckCards } from '../lib/api';

type CharacteristicOption = {
  id: DeckCharacteristicId;
  label: string;
  description: string;
};

type CardReplacementModalProps = {
  targetCard: DeckCardSuggestionWithCard;
  onClose: () => void;
  leaderCardId: string;
  onReplace: (card: OptcgCard) => void;
  excludeCardIds: string[];
  characteristicOptions: CharacteristicOption[];
  setIds: string[];
  metaOnly: boolean;
};

type SuggestionState = {
  loading: boolean;
  error: string | null;
  results: CardSuggestion[];
  characteristicId?: DeckCharacteristicId;
  lastQuery?: string;
};

export function CardReplacementModal({
  targetCard,
  onClose,
  leaderCardId,
  onReplace,
  excludeCardIds,
  characteristicOptions,
  setIds,
  metaOnly,
}: CardReplacementModalProps) {
  const [cardIdQuery, setCardIdQuery] = useState('');
  const [selectedCharacteristic, setSelectedCharacteristic] = useState<DeckCharacteristicId | ''>('');
  const [state, setState] = useState<SuggestionState>({ loading: false, error: null, results: [] });

  const canSearchById = cardIdQuery.trim().length >= 2;
  const characteristicMap = useMemo(() => {
    return characteristicOptions.reduce<Record<string, CharacteristicOption>>((acc, option) => {
      acc[option.id] = option;
      return acc;
    }, {});
  }, [characteristicOptions]);

  const runSearch = async (payload: CardSuggestionPayload, mode: 'id' | 'characteristic') => {
    try {
      setState((prev) => ({ ...prev, loading: true, error: null }));
      const response = await suggestDeckCards(payload);
      setState({
        loading: false,
        error: null,
        results: response.cards,
        characteristicId: response.characteristicId,
        lastQuery: mode === 'id' ? payload.cardIdQuery : response.characteristicId,
      });
    } catch (error) {
      console.error(error);
      setState({ loading: false, error: 'Failed to fetch card suggestions. Please try again.', results: [] });
    }
  };

  const handleSearchById = () => {
    if (!canSearchById) {
      return;
    }
    runSearch(
      {
        leaderCardId,
        cardIdQuery: cardIdQuery.trim(),
        excludeCardIds,
        limit: 10,
        setIds,
        metaOnly,
      },
      'id',
    );
  };

  const handleSearchByCharacteristic = () => {
    if (!selectedCharacteristic) {
      return;
    }
    runSearch(
      {
        leaderCardId,
        characteristicId: selectedCharacteristic,
        excludeCardIds,
        limit: 12,
        setIds,
        metaOnly,
      },
      'characteristic',
    );
  };

  const handleReplaceCard = (card: OptcgCard) => {
    onReplace(card);
    setState({ loading: false, error: null, results: [] });
    setCardIdQuery('');
    setSelectedCharacteristic('');
    onClose();
  };

  return (
    <Dialog onClose={onClose}>
      <div className="space-y-6">
        <header className="space-y-2">
          <p className="text-xs uppercase tracking-wide text-primary-300">Replace Card</p>
          <h2 className="text-2xl font-semibold text-white">{targetCard.card.name}</h2>
          <p className="text-sm text-slate-300">Current copy count: {targetCard.quantity}</p>
        </header>

        <section className="space-y-3">
          <h3 className="text-sm font-semibold text-white">Search by Card ID</h3>
          <div className="flex flex-col gap-3 md:flex-row">
            <input
              type="text"
              className="flex-1 rounded-xl border border-slate-800 bg-slate-950 px-4 py-2 text-sm text-white focus:border-primary-400 focus:outline-none focus:ring-2 focus:ring-primary-500/50"
              placeholder="e.g. OP05-060"
              value={cardIdQuery}
              onChange={(event) => setCardIdQuery(event.target.value)}
            />
            <button
              type="button"
              onClick={handleSearchById}
              disabled={!canSearchById || state.loading}
              className="rounded-xl bg-primary-500 px-4 py-2 text-sm font-semibold uppercase tracking-wide text-white shadow-lg shadow-primary-900/60 transition hover:bg-primary-400 disabled:cursor-not-allowed disabled:bg-primary-700"
            >
              Search
            </button>
          </div>
          <p className="text-xs text-slate-400">Enter at least two characters of the card ID. Results will be limited to cards that meet your leader&apos;s color identity.</p>
        </section>

        <section className="space-y-3">
          <h3 className="text-sm font-semibold text-white">Boost a Deck Characteristic</h3>
          <div className="flex flex-col gap-3 md:flex-row">
            <select
              value={selectedCharacteristic}
              onChange={(event) => setSelectedCharacteristic(event.target.value as DeckCharacteristicId)}
              className="flex-1 rounded-xl border border-slate-800 bg-slate-950 px-4 py-2 text-sm text-white focus:border-primary-400 focus:outline-none focus:ring-2 focus:ring-primary-500/50"
            >
              <option value="">Select characteristic</option>
              {characteristicOptions.map((option) => (
                <option key={option.id} value={option.id}>
                  {option.label}
                </option>
              ))}
            </select>
            <button
              type="button"
              onClick={handleSearchByCharacteristic}
              disabled={!selectedCharacteristic || state.loading}
              className="rounded-xl border border-primary-400 px-4 py-2 text-sm font-semibold uppercase tracking-wide text-primary-200 transition hover:bg-primary-500/10 disabled:cursor-not-allowed disabled:border-slate-700 disabled:text-slate-500"
            >
              Find cards
            </button>
          </div>
          {selectedCharacteristic && (
            <p className="text-xs text-slate-400">{characteristicMap[selectedCharacteristic]?.description}</p>
          )}
          {metaOnly && (
            <p className="text-xs text-primary-300">Meta-only mode enabled: only competitive cards will be suggested.</p>
          )}
        </section>

        {state.error && <p className="text-sm text-rose-400">{state.error}</p>}

        <section className="space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-white">Suggestions</h3>
            {state.loading && <span className="text-xs text-slate-400">Loading…</span>}
          </div>
          {state.results.length === 0 && !state.loading ? (
            <p className="rounded-lg border border-dashed border-slate-700 bg-slate-900/50 p-4 text-sm text-slate-400">
              Use the search above to find cards that match your criteria.
            </p>
          ) : (
            <div className="grid gap-4 md:grid-cols-2">
              {state.results.map((suggestion) => (
                <div
                  key={suggestion.card.id}
                  className="flex gap-3 rounded-xl border border-slate-800 bg-slate-950/70 p-3 shadow-inner shadow-slate-900/40"
                >
                  <div className="relative h-20 w-14 overflow-hidden rounded-lg border border-slate-800">
                    <Image
                      src={suggestion.card.imageUrl}
                      alt={suggestion.card.name}
                      fill
                      sizes="80px"
                      className="object-cover"
                    />
                  </div>
                  <div className="flex flex-1 flex-col justify-between text-sm text-slate-200">
                    <div>
                      <p className="font-semibold text-white">{suggestion.card.name}</p>
                      <p className="text-xs uppercase tracking-wide text-primary-300">{suggestion.card.id}</p>
                      <p className="mt-1 text-xs text-slate-400">
                        {suggestion.rationale ?? `Score ${suggestion.normalizedScore.toFixed(1)}`}
                      </p>
                    </div>
                    <div className="flex items-center justify-between text-xs text-slate-400">
                      <span>
                        {suggestion.card.type} • {suggestion.card.color ?? 'Colorless'}
                      </span>
                      <button
                        type="button"
                        onClick={() => handleReplaceCard(suggestion.card)}
                        className="rounded-lg bg-primary-500 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-white transition hover:bg-primary-400"
                      >
                        Replace
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>
      </div>
    </Dialog>
  );
}
