import Image from 'next/image';
import { DeckCardSuggestionWithCard } from '@shared/contracts/ai';

type DeckCardProps = {
  deckCard: DeckCardSuggestionWithCard;
  onSelect?: (deckCard: DeckCardSuggestionWithCard) => void;
  onIncrement?: (deckCard: DeckCardSuggestionWithCard) => void;
  onDecrement?: (deckCard: DeckCardSuggestionWithCard) => void;
  onReplace?: (deckCard: DeckCardSuggestionWithCard) => void;
  disableIncrement?: boolean;
  disableDecrement?: boolean;
};

export function DeckCard({
  deckCard,
  onSelect,
  onIncrement,
  onDecrement,
  onReplace,
  disableIncrement,
  disableDecrement,
}: DeckCardProps) {
  return (
    <div className="flex flex-col overflow-hidden rounded-xl border border-slate-800 bg-slate-950/60 shadow-sm shadow-slate-900/40 transition">
      <button
        type="button"
        onClick={() => onSelect?.(deckCard)}
        className="bg-slate-900/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-400/60"
      >
        <div className="relative mx-auto aspect-[2/3] w-full">
          <Image
            src={deckCard.card.imageUrl}
            alt={deckCard.card.name}
            fill
            sizes="(max-width: 768px) 50vw, 240px"
            className="object-cover"
            priority={false}
          />
          {/* REPLACE button at center, above quantity controls */}
          <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 flex flex-col items-center gap-2">
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                onReplace?.(deckCard);
              }}
              className="rounded-lg border border-primary-400 bg-black/85 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-primary-200 shadow-lg backdrop-blur-sm transition hover:bg-primary-500/10"
            >
              Replace
            </button>
            {/* Quantity controls at center */}
            <div
              className="flex items-center gap-1.5 rounded-full bg-black/85 px-2 py-1 shadow-lg backdrop-blur-sm"
              onClick={(e) => e.stopPropagation()}
            >
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  onDecrement?.(deckCard);
                }}
                disabled={disableDecrement}
                className="h-5 w-5 flex items-center justify-center rounded-full border border-white/30 bg-white/10 text-white transition hover:bg-white/20 hover:border-white/50 disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:bg-white/10"
              >
                <span className="text-xs font-bold leading-none">−</span>
              </button>
              <span className="min-w-[1.5rem] text-center text-sm font-bold text-white drop-shadow-[0_2px_4px_rgba(0,0,0,0.8)]">
                x{deckCard.quantity}
              </span>
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  onIncrement?.(deckCard);
                }}
                disabled={disableIncrement}
                className="h-5 w-5 flex items-center justify-center rounded-full border border-white/30 bg-white/10 text-white transition hover:bg-white/20 hover:border-white/50 disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:bg-white/10"
              >
                <span className="text-xs font-bold leading-none">+</span>
              </button>
            </div>
          </div>
        </div>
      </button>
      <div className="flex flex-1 flex-col gap-3 p-4 text-left text-sm text-slate-200">
        <p className="line-clamp-3 text-xs text-slate-300">{deckCard.rationale}</p>
      </div>
    </div>
  );
}
