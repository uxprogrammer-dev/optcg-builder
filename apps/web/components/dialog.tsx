import { ReactNode, useEffect } from 'react';
import { createPortal } from 'react-dom';

type DialogProps = {
  children: ReactNode;
  onClose: () => void;
};

export function Dialog({ children, onClose }: DialogProps) {
  useEffect(() => {
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onClose();
      }
    };

    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [onClose]);

  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 px-4 py-10 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
    >
      <div className="relative w-full max-w-5xl overflow-hidden rounded-2xl border border-slate-800 bg-slate-950/95 p-6 shadow-2xl shadow-slate-950/80">
        <button
          type="button"
          onClick={onClose}
          aria-label="Close"
          className="absolute right-4 top-4 rounded-full border border-slate-800 bg-slate-900/80 px-2 py-1 text-xs uppercase tracking-wide text-slate-300 transition hover:border-primary-400 hover:text-primary-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-300"
        >
          Close
        </button>
        {children}
      </div>
    </div>,
    document.body,
  );
}
