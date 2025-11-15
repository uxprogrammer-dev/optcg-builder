import Image from 'next/image';
import { DeckCardSuggestionWithCard } from '@shared/contracts/ai';
import { Dialog } from './dialog';

type CardDetailModalProps = {
  deckCard: DeckCardSuggestionWithCard;
  onClose: () => void;
};

export function CardDetailModal({ deckCard, onClose }: CardDetailModalProps) {
  return (
    <Dialog onClose={onClose}>
      <div className="grid gap-8 md:grid-cols-[minmax(0,400px)_1fr]">
        <div className="relative mx-auto h-[28rem] w-80 overflow-hidden rounded-xl shadow-lg shadow-slate-900/60 md:h-[40rem] md:w-96">
          <Image
            src={deckCard.card.imageUrl}
            alt={deckCard.card.name}
            fill
            sizes="(max-width: 768px) 50vw, 384px"
            className="object-contain"
            priority
          />
        </div>

        <div className="space-y-4">
          <header className="space-y-2">
            <p className="text-xs uppercase tracking-wide text-primary-300">
              {deckCard.card.id} • {deckCard.role}
            </p>
            <h3 className="text-2xl font-semibold text-white">{deckCard.card.name}</h3>
            <p className="text-sm text-slate-300">
              {deckCard.card.type} • {deckCard.card.color ?? 'Colorless'}{' '}
              {deckCard.card.cost ? `• Cost ${deckCard.card.cost}` : ''}
            </p>
          </header>

          <section className="space-y-3 text-sm leading-relaxed text-slate-200">
            <div>
              <h4 className="text-sm font-semibold text-primary-200">Why this card?</h4>
              <p className="mt-1 whitespace-pre-wrap text-slate-300">
                {deckCard.rationale || 'Supports the selected leader strategy.'}
              </p>
            </div>

            {deckCard.card.text && (
              <div>
                <h4 className="text-sm font-semibold text-primary-200">Card Text</h4>
                <p className="mt-1 whitespace-pre-wrap text-slate-300">{deckCard.card.text}</p>
              </div>
            )}

            <div className="flex flex-wrap gap-3 text-xs uppercase tracking-wide text-slate-400">
              {deckCard.card.subtypes.length > 0 && (
                <span className="rounded-full border border-slate-800 px-3 py-1">
                  {deckCard.card.subtypes.join(' • ')}
                </span>
              )}
              {deckCard.card.power && (
                <span className="rounded-full border border-slate-800 px-3 py-1">
                  Power {deckCard.card.power}
                </span>
              )}
              {deckCard.card.life && (
                <span className="rounded-full border border-slate-800 px-3 py-1">
                  Life {deckCard.card.life}
                </span>
              )}
            </div>
          </section>

          <footer className="flex items-center justify-between text-xs text-slate-400">
            <span>Set: {deckCard.card.setName}</span>
            <span className="font-semibold text-primary-300">x{deckCard.quantity}</span>
          </footer>
        </div>
      </div>
    </Dialog>
  );
}
