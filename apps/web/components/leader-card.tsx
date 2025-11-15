import Image from 'next/image';
import clsx from 'clsx';
import { LeaderSuggestionWithCard } from '@shared/contracts/ai';

type LeaderCardProps = {
  leader: LeaderSuggestionWithCard;
  isSelected: boolean;
  onSelect: (leader: LeaderSuggestionWithCard) => void;
};

export function LeaderCard({ leader, isSelected, onSelect }: LeaderCardProps) {
  return (
    <button
      type="button"
      onClick={() => onSelect(leader)}
      className={clsx(
        'flex flex-col overflow-hidden rounded-xl border transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-400 hover:border-primary-500',
        isSelected ? 'border-primary-500 shadow-lg shadow-primary-900/40' : 'border-slate-800',
      )}
    >
      <div className="bg-slate-900/60">
        <div className="relative mx-auto aspect-[2/3] w-full">
          <Image
            src={leader.card.imageUrl}
            alt={leader.card.name}
            fill
            sizes="(max-width: 768px) 50vw, 260px"
            className="object-cover"
            priority
          />
        </div>
      </div>
      <div className="bg-slate-900 p-4 text-left text-sm text-slate-300 space-y-2">
        <div>
          <h3 className="text-lg font-semibold text-white">{leader.card.name}</h3>
          <p className="text-xs uppercase tracking-wide text-primary-300">{leader.card.id}</p>
        </div>
        <p className="text-sm leading-relaxed text-slate-200">
          {leader.strategySummary ?? leader.rationale}
        </p>
        <p className="text-xs text-slate-400 leading-relaxed">
          {leader.rationale}
        </p>
      </div>
    </button>
  );
}
