'use client';

import { ChangeEvent, FormEvent, useEffect, useMemo, useRef, useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import {
  downloadDeckArchive,
  fetchLeaderSuggestions,
  generateDeck,
  rerunDeckReview,
  fetchSetOptions,
  fetchProgress,
  SetOption,
} from '../lib/api';
import { LeaderCard } from '../components/leader-card';
import { DeckCard } from '../components/deck-card';
import { DeckGenerationResponse, LeaderSuggestionsResponse } from '../lib/api';
import { DeckCardSuggestionWithCard, LeaderSuggestionWithCard } from '@shared/contracts/ai';
import { OptcgCard } from '@shared/types/optcg-card';
import { CardDetailModal } from '../components/card-detail-modal';
import { DeckRadarChart } from '../components/deck-radar-chart';
import { CardReplacementModal } from '../components/card-replacement-modal';

const CHARACTERISTIC_CONFIGS = [
  {
    id: 'control',
    label: 'Control',
    description: 'Ability to manage the opponent\'s board state and dictate tempo.',
    keywords: ['control', 'board', 'active', 'rest', 'tap', 'set up', 'stun', 'lock'],
  },
  {
    id: 'cardAdvantage',
    label: 'Card Advantage',
    description: 'Drawing or searching effects that increase hand resources.',
    keywords: ['draw', 'search', 'add to hand', 'look at', 'reveal', 'tutor'],
  },
  {
    id: 'removal',
    label: 'Removal',
    description: 'Tools that eliminate or disable opposing cards.',
    keywords: ['destroy', 'k.o.', 'remove', 'banish', 'bounce', 'retire', 'return to hand', 'bottom of deck'],
  },
  {
    id: 'trashManipulation',
    label: 'Trash Manipulation',
    description: 'Effects that interact with trash, discard piles, or bottom-deck zones.',
    keywords: ['trash', 'graveyard', 'discard pile', 'from trash', 'from your trash', 'bottom of your deck'],
  },
  {
    id: 'aggression',
    label: 'Aggression',
    description: 'Fast offensive pressure, power boosts, and repeated attacks.',
    keywords: ['attack', 'damage', 'double attack', 'rush', '+', 'on attack', 'power up'],
  },
  {
    id: 'defense',
    label: 'Defense',
    description: 'Blocking, protection, and effects that preserve life totals.',
    keywords: ['blocker', 'counter', 'life', 'prevent', 'reduce damage', 'guard', 'shield', 'heal', 'cannot be k.o.'],
  },
  {
    id: 'synergy',
    label: 'Synergy',
    description: 'Combo potential and conditional interactions between cards.',
    keywords: ['when', 'if', 'combo', 'trigger', 'activate', 'support', 'ally', 'partner'],
  },
  {
    id: 'economy',
    label: 'Economy',
    description: 'Resource generation and DON cost efficiency tools.',
    keywords: ['cost', 'reduce cost', 'don', 'don!!', 'gain don', 'set don', 'resource'],
  },
] as const;

type CharacteristicId = (typeof CHARACTERISTIC_CONFIGS)[number]['id'];

type DeckCharacteristic = {
  id: CharacteristicId;
  label: string;
  description: string;
  score: number;
};

type CharacteristicScores = Record<CharacteristicId, number>;

const QUANTITY_MULTIPLIERS: Record<number, number> = {
  1: 0.5,
  2: 0.75,
  3: 1,
  4: 1.25,
};

function getQuantityMultiplier(quantity: number): number {
  if (quantity >= 4) return QUANTITY_MULTIPLIERS[4];
  if (quantity === 3) return QUANTITY_MULTIPLIERS[3];
  if (quantity === 2) return QUANTITY_MULTIPLIERS[2];
  return QUANTITY_MULTIPLIERS[1];
}

function countKeywordHits(text: string, keyword: string): number {
  const escaped = keyword.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const regex = new RegExp(escaped, 'g');
  return (text.match(regex) ?? []).length;
}

function extractBountyFromText(text: string): number {
  // Look for bounty patterns like "bounty: 1000000" or "bounty 1000000" or numbers after "bounty"
  const bountyPatterns = [
    /bounty[:\s]+(\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?)/i,
    /bounty[:\s]+(\d+)/i,
  ];
  
  for (const pattern of bountyPatterns) {
    const match = text.match(pattern);
    if (match && match[1]) {
      // Remove commas and spaces, then parse
      const bountyStr = match[1].replace(/[,\s]/g, '');
      const bounty = parseInt(bountyStr, 10);
      if (!isNaN(bounty) && bounty > 0) {
        return bounty;
      }
    }
  }
  
  return 0;
}

function computeBountyScore(cards: DeckCardSuggestionWithCard[]): number {
  let totalBounty = 0;
  
  cards.forEach((card) => {
    const cardText = `${card.card.name ?? ''} ${card.card.text ?? ''}`;
    const bounty = extractBountyFromText(cardText);
    // Multiply by quantity since each copy contributes to the total
    totalBounty += bounty * card.quantity;
  });
  
  return totalBounty;
}

function analyzeDeckCharacteristics(cards: DeckCardSuggestionWithCard[]): DeckCharacteristic[] {
  const lowerCaseCards = cards.map((card) => ({
    quantity: card.quantity,
    role: card.role,
    text: `${card.card.name ?? ''} ${card.card.text ?? ''}`.toLowerCase(),
  }));

  const baseScores: CharacteristicScores = {
    control: 0,
    cardAdvantage: 0,
    removal: 0,
    trashManipulation: 0,
    aggression: 0,
    defense: 0,
    synergy: 0,
    economy: 0,
  };

  lowerCaseCards.forEach(({ quantity, role, text }) => {
    const quantityMultiplier = getQuantityMultiplier(quantity);

    CHARACTERISTIC_CONFIGS.forEach(({ id, keywords }) => {
      let hits = 0;
      keywords.forEach((keyword) => {
        hits += countKeywordHits(text, keyword);
      });

      if (id === 'aggression' && role === 'character' && text.includes('rush')) {
        hits += 1;
      }
      if (id === 'defense' && role === 'event' && text.includes('counter')) {
        hits += 1;
      }
      if (id === 'economy' && text.includes('don')) {
        hits += 1;
      }

      if (hits > 0) {
        baseScores[id] += hits * quantityMultiplier;
      }
    });
  });

  return CHARACTERISTIC_CONFIGS.map(({ id, label, description }) => {
    const rawScore = baseScores[id];
    const normalized = Math.min(10, Number(((1 - Math.exp(-rawScore / 6)) * 10).toFixed(1)));
    return { id, label, description, score: normalized };
  });
}

function resolveCardRole(card: OptcgCard): DeckCardSuggestionWithCard['role'] {
  const type = card.type?.toLowerCase() ?? '';
  if (type.includes('character')) return 'character';
  if (type.includes('event')) return 'event';
  if (type.includes('stage')) return 'stage';
  if (type.includes('don')) return 'don';
  if (type.includes('counter')) return 'counter';
  return 'other';
}

function getBaseCardCode(cardId: string): string {
  const match = cardId.match(/^([A-Z0-9]+-[0-9]+)/i);
  return match ? match[1].toUpperCase() : cardId.toUpperCase();
}

function FormattedReview({ text }: { text: string }) {
  // Helper function to render text with markdown bold formatting
  const renderFormattedText = (text: string) => {
    // Replace **text** with <strong>text</strong>
    const parts: (string | React.ReactElement)[] = [];
    const boldRegex = /\*\*(.+?)\*\*/g;
    let lastIndex = 0;
    let match;
    let key = 0;

    while ((match = boldRegex.exec(text)) !== null) {
      // Add text before the match
      if (match.index > lastIndex) {
        parts.push(text.substring(lastIndex, match.index));
      }
      // Add bold text
      parts.push(
        <strong key={key++} className="font-semibold text-white">
          {match[1]}
        </strong>,
      );
      lastIndex = match.index + match[0].length;
    }
    // Add remaining text
    if (lastIndex < text.length) {
      parts.push(text.substring(lastIndex));
    }

    return parts.length > 0 ? parts : text;
  };

  // Split text into paragraphs and format markdown-style syntax
  const formatReview = (reviewText: string) => {
    const lines = reviewText.split('\n');
    const elements: React.ReactNode[] = [];
    let currentParagraph: string[] = [];
    let listItems: string[] = [];
    let inList = false;

    const flushParagraph = () => {
      if (currentParagraph.length > 0) {
        const paraText = currentParagraph.join(' ').trim();
        if (paraText) {
          elements.push(
            <p key={`para-${elements.length}`} className="mb-4 text-sm leading-relaxed text-slate-200 last:mb-0">
              {renderFormattedText(paraText)}
            </p>,
          );
        }
        currentParagraph = [];
      }
    };

    const flushList = () => {
      if (listItems.length > 0) {
        elements.push(
          <ul key={`list-${elements.length}`} className="mb-4 ml-4 list-disc space-y-2 text-sm text-slate-200 last:mb-0">
            {listItems.map((item, idx) => (
              <li key={idx} className="leading-relaxed">
                {renderFormattedText(item.trim())}
              </li>
            ))}
          </ul>,
        );
        listItems = [];
        inList = false;
      }
    };

    lines.forEach((line, idx) => {
      const trimmed = line.trim();

      // Check for markdown headings: # Heading, ## Subheading, ### Sub-subheading
      const headingMatch = trimmed.match(/^(#{1,3})\s+(.+)$/);
      if (headingMatch) {
        flushList();
        flushParagraph();
        const [, hashes, title] = headingMatch;
        const level = hashes.length;
        // Remove any markdown bold from title
        const cleanTitle = title.replace(/\*\*/g, '');
        if (level === 1) {
          elements.push(
            <h3 key={`h1-${idx}`} className="mb-3 mt-6 text-lg font-semibold text-primary-300 first:mt-0">
              {cleanTitle}
            </h3>,
          );
        } else if (level === 2) {
          elements.push(
            <h4 key={`h2-${idx}`} className="mb-2 mt-4 text-base font-semibold text-primary-300">
              {cleanTitle}
            </h4>,
          );
        } else {
          elements.push(
            <h5 key={`h3-${idx}`} className="mb-2 mt-3 text-sm font-semibold text-primary-200">
              {cleanTitle}
            </h5>,
          );
        }
        return;
      }

      // Check for numbered sections (e.g., "1. Title", "2. Title")
      const numberedSectionMatch = trimmed.match(/^(\d+)\.\s+(.+)$/);
      if (numberedSectionMatch) {
        flushList();
        flushParagraph();
        const [, number, title] = numberedSectionMatch;
        // Only treat as section if it's short (likely a heading)
        if (title.length < 80) {
          // Remove markdown bold from title
          const cleanTitle = title.replace(/\*\*/g, '');
          elements.push(
            <h4 key={`section-${number}`} className="mb-3 mt-6 text-base font-semibold text-primary-300 first:mt-0">
              {number}. {cleanTitle}
            </h4>,
          );
          return;
        }
      }

      // Check for bold headings (lines that are short and end with colon)
      // Also check if it starts with ** (markdown bold)
      if (trimmed.length < 60 && (trimmed.endsWith(':') || trimmed.match(/^\*\*.+:\*\*$/))) {
        flushList();
        flushParagraph();
        // Remove markdown bold and colon
        const cleanHeading = trimmed.replace(/\*\*/g, '').replace(/:$/, '');
        elements.push(
          <h4 key={`heading-${idx}`} className="mb-2 mt-4 text-sm font-semibold text-primary-200 first:mt-0">
            {cleanHeading}
          </h4>,
        );
        return;
      }

      // Check for list items (lines starting with "- ", "• ", "* ", or markdown-style)
      const listItemMatch = trimmed.match(/^[-•*]\s+(.+)$/);
      if (listItemMatch) {
        flushParagraph();
        inList = true;
        listItems.push(listItemMatch[1]);
        return;
      }

      // Check for numbered list items (e.g., "1. Item", "2. Item")
      // Only if it's not too long (likely a list item, not a section)
      const numberedListItemMatch = trimmed.match(/^\d+\.\s+(.+)$/);
      if (numberedListItemMatch) {
        const itemText = numberedListItemMatch[1];
        // Treat as list item if it's part of a list or if it's short
        if (inList || itemText.length < 150) {
          flushParagraph();
          inList = true;
          listItems.push(itemText);
          return;
        }
      }

      // Regular paragraph text
      if (trimmed) {
        flushList();
        currentParagraph.push(trimmed);
      } else {
        // Empty line - flush current paragraph
        flushParagraph();
        inList = false; // Reset list state on empty line
      }
    });

    flushList();
    flushParagraph();

    return elements.length > 0 ? elements : <p className="text-sm leading-relaxed text-slate-200">{text}</p>;
  };

  return <div className="space-y-2">{formatReview(text)}</div>;
}

export default function HomePage() {
  const [prompt, setPrompt] = useState('');
  const [useOpenAi, setUseOpenAi] = useState(false);
  const [leaders, setLeaders] = useState<LeaderSuggestionWithCard[]>([]);
  const [selectedLeader, setSelectedLeader] = useState<LeaderSuggestionWithCard | null>(null);
  const [deck, setDeck] = useState<DeckGenerationResponse | null>(null);
  const [selectedDeckCard, setSelectedDeckCard] = useState<DeckCardSuggestionWithCard | null>(null);
  const [replacementTarget, setReplacementTarget] = useState<DeckCardSuggestionWithCard | null>(null);
  const [availableSets, setAvailableSets] = useState<SetOption[]>([]);
  const [selectedSetIds, setSelectedSetIds] = useState<string[]>([]);
  const [metaOnlyFilter, setMetaOnlyFilter] = useState(false);
  const [tournamentOnlyFilter, setTournamentOnlyFilter] = useState(false);
  const [reviewLoading, setReviewLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [progressLabel, setProgressLabel] = useState('');
  const [progressVisible, setProgressVisible] = useState(false);
  const [progressPhase, setProgressPhase] = useState<'leaders' | 'deck' | null>(null);
  const [currentProgressId, setCurrentProgressId] = useState<string | null>(null);
  const progressTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const progressHideRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const progressPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  
  // Cards per row (3-6), persisted to localStorage
  const [cardsPerRow, setCardsPerRow] = useState(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('deck-cards-per-row');
      if (saved) {
        const value = parseInt(saved, 10);
        if (value >= 3 && value <= 6) {
          return value;
        }
      }
    }
    return 3; // default
  });

  const clearProgressTimer = () => {
    if (progressTimerRef.current) {
      clearInterval(progressTimerRef.current);
      progressTimerRef.current = null;
    }
  };

  const clearHideTimeout = () => {
    if (progressHideRef.current) {
      clearTimeout(progressHideRef.current);
      progressHideRef.current = null;
    }
  };

  const clearProgressPoll = () => {
    if (progressPollRef.current) {
      clearInterval(progressPollRef.current);
      progressPollRef.current = null;
    }
  };

  const startProgressPolling = (progressId: string, phase: 'leaders' | 'deck') => {
    clearProgressPoll();
    clearProgressTimer(); // Stop fake timer when real polling starts
    setCurrentProgressId(progressId);
    
    console.log(`[Progress] Starting polling for ${phase} with progressId: ${progressId}`);
    
    // Poll function
    const pollProgress = async () => {
      try {
        const update = await fetchProgress(progressId);
        console.log(`[Progress] Received update:`, { phase: update.phase, step: update.step, progress: update.progress });
        
        // Handle "Not found" case - progress might have been cleaned up
        if (update.step === 'Not found' || (update.phase === null && update.progress === 0)) {
          console.log(`[Progress] Progress not found or already completed`);
          // Progress might have completed before we started polling
          // Don't clear polling yet, keep trying a few more times
          return;
        }
        
        if (update.phase === phase || update.phase === null) {
          setProgress(update.progress);
          setProgressLabel(update.step);
          if (update.progress >= 100) {
            console.log(`[Progress] Progress complete at 100%`);
            clearProgressPoll();
            setCurrentProgressId(null);
            finishProgress(phase, update.step || `${phase === 'leaders' ? 'Leader suggestions' : 'Deck list'} ready!`);
          }
        } else {
          console.log(`[Progress] Phase mismatch: expected ${phase}, got ${update.phase}`);
        }
      } catch (error) {
        console.error('[Progress] Failed to fetch progress:', error);
        // Continue polling even on error
      }
    };
    
    // Poll immediately first, then every 500ms
    pollProgress();
    progressPollRef.current = setInterval(pollProgress, 500);
  };

  const startProgress = (phase: 'leaders' | 'deck') => {
    clearProgressTimer();
    clearHideTimeout();
    setProgressPhase(phase);
    setProgressVisible(true);
    setProgress(phase === 'deck' ? 20 : 8);
    setProgressLabel(
      phase === 'leaders' ? 'Initializing leader suggestions...' : 'Initializing deck generation...',
    );
    const cap = phase === 'leaders' ? 82 : 94;
    progressTimerRef.current = setInterval(() => {
      setProgress((previous) => {
        if (previous >= cap) {
          return previous;
        }
        const increment = phase === 'leaders' ? 6 : 5;
        return Math.min(cap, previous + increment);
      });
    }, 450);
  };

  const finishProgress = (phase: 'leaders' | 'deck', message: string) => {
    if (progressPhase !== phase) {
      return;
    }
    clearProgressTimer();
    clearHideTimeout();
    setProgressLabel(message);
    setProgress(100);
    progressHideRef.current = setTimeout(() => {
      setProgressVisible(false);
      setProgress(0);
      setProgressPhase(null);
    }, 800);
  };

  const failProgress = (phase: 'leaders' | 'deck', message: string) => {
    if (progressPhase !== phase) {
      return;
    }
    clearProgressTimer();
    clearHideTimeout();
    setProgressLabel(message);
    setProgress(100);
    progressHideRef.current = setTimeout(() => {
      setProgressVisible(false);
      setProgress(0);
      setProgressPhase(null);
    }, 1400);
  };

  useEffect(
    () => () => {
      clearProgressTimer();
      clearHideTimeout();
      clearProgressPoll();
    },
    [],
  );

  useEffect(() => {
    let cancelled = false;
    fetchSetOptions()
      .then((options) => {
        if (!cancelled) {
          setAvailableSets(options);
        }
      })
      .catch((error) => {
        console.error('Failed to load set options', error);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const leaderMutation = useMutation({
    mutationFn: async ({ promptText, regenerate }: { promptText: string; regenerate?: boolean }) => {
      setErrorMessage(null);
      // Generate progressId on frontend so we can start polling immediately
      const frontendProgressId = `leaders-${Date.now()}-${Math.random().toString(36).substring(7)}`;
      
      // Start polling immediately with the predicted progressId
      // The backend will use this same ID format, so it should match
      startProgressPolling(frontendProgressId, 'leaders');
      
      const response = await fetchLeaderSuggestions(promptText, {
        useOpenAi,
        setIds: selectedSetIds,
        metaOnly: metaOnlyFilter,
        tournamentOnly: tournamentOnlyFilter,
        regenerate,
        progressId: frontendProgressId, // Send our progressId to backend
      });
      return response;
    },
    onMutate: () => {
      startProgress('leaders');
    },
    onSuccess: (data: LeaderSuggestionsResponse) => {
      console.log('[LeaderMutation] Success response:', { hasProgressId: !!data.progressId, progressId: data.progressId });
      // Polling already started in mutationFn, just verify it's still running
      // If progressId doesn't match, switch to the one from response
      if (data.progressId && currentProgressId !== data.progressId) {
        console.log('[LeaderMutation] ProgressId mismatch, switching to response ID:', data.progressId);
        clearProgressPoll();
        startProgressPolling(data.progressId, 'leaders');
      }
      // Don't clear polling here - let it continue until progress reaches 100%
      setLeaders(data.leaders);
      setSelectedLeader(null);
      setDeck(null);
    },
    onError: (error: unknown) => {
      console.error(error);
      setErrorMessage('Failed to generate leader suggestions. Please try again.');
      failProgress('leaders', 'Could not generate leader suggestions.');
    },
  });

  const deckMutation = useMutation({
    mutationFn: async (leader: LeaderSuggestionWithCard) => {
      setErrorMessage(null);
      // Generate progressId on frontend so we can start polling immediately
      const frontendProgressId = `deck-${Date.now()}-${Math.random().toString(36).substring(7)}`;
      
      // Start polling immediately with the predicted progressId
      startProgressPolling(frontendProgressId, 'deck');
      
      const response = await generateDeck(prompt, leader.card.id, {
        useOpenAi,
        setIds: selectedSetIds,
        metaOnly: metaOnlyFilter,
        tournamentOnly: tournamentOnlyFilter,
        progressId: frontendProgressId, // Send our progressId to backend
      });
      return response;
    },
    onMutate: () => {
      startProgress('deck');
    },
    onSuccess: (data: DeckGenerationResponse) => {
      console.log('[DeckMutation] Success response:', { hasProgressId: !!data.progressId, progressId: data.progressId });
      // Polling already started in mutationFn, just verify it's still running
      // If progressId doesn't match, switch to the one from response
      if (data.progressId && currentProgressId !== data.progressId) {
        console.log('[DeckMutation] ProgressId mismatch, switching to response ID:', data.progressId);
        clearProgressPoll();
        startProgressPolling(data.progressId, 'deck');
      }
      // Don't clear polling here - let it continue until progress reaches 100%
      setDeck(data);
      setSelectedDeckCard(null);
    },
    onError: (error: unknown) => {
      console.error(error);
      setErrorMessage('Failed to build deck. Please choose another leader or try again.');
      failProgress('deck', 'Deck build failed.');
    },
  });

  const totalQuantity = useMemo(
    () => deck?.cards.reduce((sum, c) => sum + c.quantity, 0) ?? 0,
    [deck],
  );

  const replacementExcludeIds = useMemo(
    () =>
      deck
        ? deck.cards
            .filter((card) => !replacementTarget || card.card.id !== replacementTarget.card.id)
            .map((card) => card.card.id)
        : [],
    [deck, replacementTarget],
  );

  const deckCharacteristics = useMemo(
    () => (deck ? analyzeDeckCharacteristics(deck.cards) : null),
    [deck],
  );

  const selectedSetLabels = useMemo(
    () =>
      selectedSetIds.map((id) => availableSets.find((set) => set.id === id)?.label ?? id),
    [selectedSetIds, availableSets],
  );

  const deckTotalStatusClass = totalQuantity === 50 ? 'text-slate-400' : totalQuantity > 50 ? 'text-rose-400' : 'text-amber-300';
  const deckTotalHelperText =
    totalQuantity === 50
      ? 'Deck size complete (50 cards + leader).'
      : totalQuantity > 50
      ? `${totalQuantity - 50} card(s) over the 50-card limit.`
      : `${50 - totalQuantity} card(s) needed to reach 50.`;

  const busy = leaderMutation.isPending || deckMutation.isPending;

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!prompt.trim()) {
      setErrorMessage('Please provide a prompt describing your desired deck.');
      return;
    }
    leaderMutation.mutate({ promptText: prompt.trim() });
  };

  const handleSetSelectionChange = (event: ChangeEvent<HTMLSelectElement>) => {
    const values = Array.from(event.target.selectedOptions).map((option) => option.value.toUpperCase());
    setSelectedSetIds(values);
  };

  const handleRegenerateLeaders = () => {
    if (!prompt.trim() || leaderMutation.isPending) {
      return;
    }
    leaderMutation.mutate({ promptText: prompt.trim(), regenerate: true });
  };

  const handleLeaderSelect = (leader: LeaderSuggestionWithCard) => {
    setSelectedLeader(leader);
    setDeck(null);
    deckMutation.mutate(leader);
  };

  const handleDownload = async () => {
    if (!deck) {
      return;
    }
    try {
      const blob = await downloadDeckArchive(deck);
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `optcg-deck-${deck.deckId}.zip`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error(error);
      setErrorMessage('Unable to download deck images. Please try again.');
    }
  };

  const handleDownloadList = () => {
    if (!deck) {
      return;
    }

    const lines = [
      // Include the leader as 1 copy at the top of the list
      `1x${deck.leader.id}`,
      ...deck.cards.map((deckCard) => `${deckCard.quantity}x${deckCard.card.id}`.trim()),
    ];

    const blob = new Blob([lines.join('\n')], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `optcg-deck-${deck.deckId}.txt`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  };

  const handleIncrementCard = (target: DeckCardSuggestionWithCard) => {
    if (!deck) {
      return;
    }

    let updatedDeck: DeckGenerationResponse | null = null;

    setDeck((prev) => {
      if (!prev) {
        return prev;
      }
      const total = prev.cards.reduce((sum, card) => sum + card.quantity, 0);
      if (total >= 50) {
        return prev;
      }

      let updated = false;
      const cards = prev.cards.map((card) => {
        if (card.card.id !== target.card.id) {
          return card;
        }
        if (card.quantity >= 4) {
          return card;
        }
        updated = true;
        return { ...card, quantity: card.quantity + 1 };
      });

      if (!updated) {
        return prev;
      }

      const newTotalCards = cards.reduce((sum, card) => sum + card.quantity, 0);
      const nextDeck = { ...prev, cards, totalCards: newTotalCards };
      updatedDeck = nextDeck;
      return nextDeck;
    });

    if (updatedDeck) {
      setSelectedDeckCard((current) => {
        if (!current) {
          return current;
        }
        const updated = updatedDeck?.cards.find((card) => card.card.id === current.card.id);
        return updated ?? null;
      });
    }
  };

  const handleDecrementCard = (target: DeckCardSuggestionWithCard) => {
    if (!deck) {
      return;
    }

    let updatedDeck: DeckGenerationResponse | null = null;

    setDeck((prev) => {
      if (!prev) {
        return prev;
      }

      let updated = false;
      const cards = prev.cards
        .map((card) => {
          if (card.card.id !== target.card.id) {
            return card;
          }
          if (card.quantity <= 0) {
            return card;
          }
          updated = true;
          return { ...card, quantity: card.quantity - 1 };
        })
        .filter((card) => card.quantity > 0);

      if (!updated) {
        return prev;
      }

      const newTotalCards = cards.reduce((sum, card) => sum + card.quantity, 0);
      const nextDeck = { ...prev, cards, totalCards: newTotalCards };
      updatedDeck = nextDeck;
      return nextDeck;
    });

    if (updatedDeck) {
      setSelectedDeckCard((current) => {
        if (!current) {
          return current;
        }
        const updated = updatedDeck?.cards.find((card) => card.card.id === current.card.id);
        return updated ?? null;
      });
    }
  };

  const handleReplaceCard = (newCard: OptcgCard) => {
    if (!deck || !replacementTarget) {
      return;
    }

    let updatedDeck: DeckGenerationResponse | null = null;
    const targetBase = getBaseCardCode(replacementTarget.card.id);

    setDeck((prev) => {
      if (!prev) {
        return prev;
      }

      const cards = prev.cards.map((card) => ({ ...card }));
      const targetIndex = cards.findIndex((card) => card.card.id === replacementTarget.card.id);
      if (targetIndex === -1) {
        return prev;
      }

      const targetQuantity = cards[targetIndex].quantity;
      const newBase = getBaseCardCode(newCard.id);
      const newRole = resolveCardRole(newCard);
      const duplicateIndex = cards.findIndex(
        (card, index) => index !== targetIndex && getBaseCardCode(card.card.id) === newBase,
      );

      if (duplicateIndex >= 0) {
        const duplicateCard = cards[duplicateIndex];
        const combinedQuantity = Math.min(4, duplicateCard.quantity + targetQuantity);
        cards.splice(targetIndex, 1);
        const adjustedIndex = duplicateIndex > targetIndex ? duplicateIndex - 1 : duplicateIndex;
        cards[adjustedIndex] = { ...duplicateCard, quantity: combinedQuantity };
      } else {
        cards[targetIndex] = {
          ...cards[targetIndex],
          cardSetId: newCard.id,
          card: newCard,
          role: newRole,
          quantity: Math.min(targetQuantity, 4),
          rationale: 'User-selected replacement card',
        };
      }

      const newTotalCards = cards.reduce((sum, card) => sum + card.quantity, 0);
      const nextDeck = { ...prev, cards, totalCards: newTotalCards };
      updatedDeck = nextDeck;
      return nextDeck;
    });

    if (updatedDeck) {
      setSelectedDeckCard((current) => {
        if (!current) {
          return current;
        }
        if (getBaseCardCode(current.card.id) === targetBase) {
          const updated = updatedDeck?.cards.find((card) => getBaseCardCode(card.card.id) === targetBase);
          return updated ?? null;
        }
        const updated = updatedDeck?.cards.find((card) => card.card.id === current.card.id);
        return updated ?? null;
      });
    }

    setReplacementTarget(null);
  };

  const handleRerunDeckReview = async () => {
    if (!deck || totalQuantity !== 50 || reviewLoading) {
      return;
    }

    try {
      setReviewLoading(true);
      setErrorMessage(null);
      const response = await rerunDeckReview({
        prompt,
        leaderCardId: deck.leader.id,
        cards: deck.cards.map((card) => ({ cardSetId: card.card.id, quantity: card.quantity })),
      });
      setDeck((prev) => (prev ? { ...prev, deckReview: response.deckReview } : prev));
    } catch (error) {
      console.error(error);
      setErrorMessage('Unable to re-run deck review. Please try again.');
    } finally {
      setReviewLoading(false);
    }
  };

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-6xl flex-col gap-10 px-6 py-12">
      <header className="space-y-4 text-center">
        <p className="text-sm uppercase tracking-[0.4em] text-primary-400">One Piece TCG</p>
        <h1 className="text-4xl font-bold text-white md:text-5xl">AI-Powered Deck Builder</h1>
        <p className="mx-auto max-w-3xl text-base text-slate-300 md:text-lg">
          Describe the strategy, colors, or characters you want to build around. We will suggest
          competitive leaders, craft a deck list, show live card images, and let you download them
          in a single archive.
        </p>
      </header>

      <section className="rounded-2xl border border-slate-800 bg-slate-950/60 p-6 shadow-xl shadow-slate-900/40">
        <form className="flex flex-col gap-4 md:flex-row" onSubmit={handleSubmit}>
          <label className="flex-1">
            <span className="sr-only">Deck prompt</span>
            <textarea
              className="h-28 w-full resize-none rounded-xl border border-slate-800 bg-slate-950 px-4 py-3 text-base text-white shadow-inner shadow-slate-900/50 focus:border-primary-400 focus:outline-none focus:ring-2 focus:ring-primary-500/50 md:h-24"
              placeholder="Example: Aggressive red/green Straw Hat deck focusing on Rush characters and DON ramp."
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              disabled={leaderMutation.isPending}
            />
          </label>
          <div className="flex flex-col gap-2">
            <button
              type="submit"
              className="h-12 rounded-xl bg-primary-500 px-6 text-sm font-semibold uppercase tracking-wide text-white shadow-lg shadow-primary-900/60 transition hover:bg-primary-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-200 disabled:cursor-not-allowed disabled:bg-primary-700"
              disabled={leaderMutation.isPending}
            >
              {leaderMutation.isPending ? 'Generating...' : 'Suggest Leaders'}
            </button>
          </div>
        </form>
        <div className="mt-4 grid gap-4 md:grid-cols-[2fr_1fr]">
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs text-slate-400">
              <span>Select card sets</span>
              <span>{selectedSetIds.length} selected</span>
            </div>
            <select
              multiple
              value={selectedSetIds}
              onChange={handleSetSelectionChange}
              className="h-32 w-full rounded-xl border border-slate-800 bg-slate-950 px-3 py-2 text-sm text-white focus:border-primary-400 focus:outline-none focus:ring-2 focus:ring-primary-500/50"
            >
              {availableSets.map((set) => (
                <option key={set.id} value={set.id}>
                  {set.id} - {set.label || set.id}
                </option>
              ))}
            </select>
            <p className="text-xs text-slate-500">Hold Ctrl (Cmd on Mac) to select multiple sets. Leave empty to allow all sets.</p>
          </div>
          <div className="space-y-3">
            <label className="flex items-center gap-2 text-xs text-slate-400">
              <input
                type="checkbox"
                checked={useOpenAi}
                onChange={(e) => setUseOpenAi(e.target.checked)}
                disabled={leaderMutation.isPending || deckMutation.isPending}
                className="h-4 w-4 rounded border-slate-700 bg-slate-900 text-primary-500 focus:ring-2 focus:ring-primary-500/50"
              />
              <span>Use OpenAI when available</span>
            </label>
            <label className="flex items-center gap-2 text-xs text-slate-400">
              <input
                type="checkbox"
                checked={metaOnlyFilter}
                onChange={(event) => setMetaOnlyFilter(event.target.checked)}
                className="h-4 w-4 rounded border-slate-700 bg-slate-900 text-primary-500 focus:ring-2 focus:ring-primary-500/50"
              />
              <span>Meta cards only</span>
            </label>
            <label className="flex items-center gap-2 text-xs text-slate-400">
              <input
                type="checkbox"
                checked={tournamentOnlyFilter}
                onChange={(event) => setTournamentOnlyFilter(event.target.checked)}
                className="h-4 w-4 rounded border-slate-700 bg-slate-900 text-primary-500 focus:ring-2 focus:ring-primary-500/50"
              />
              <span>Use only tournament decks</span>
            </label>
          </div>
        </div>
        {errorMessage && <p className="mt-3 text-sm text-rose-400">{errorMessage}</p>}
      </section>

      {progressVisible && (
        <section className="rounded-2xl border border-primary-900/40 bg-slate-950/70 p-5 shadow-lg shadow-primary-900/40 transition">
          <div className="flex items-center justify-between text-sm text-slate-300">
            <span>{progressLabel}</span>
            <span className="font-semibold text-primary-300">{Math.round(progress)}%</span>
          </div>
          <div className="mt-3 h-2 w-full rounded-full bg-slate-800/80">
            <div
              className="h-full rounded-full bg-primary-500 transition-[width] duration-300 ease-out"
              style={{ width: `${Math.min(progress, 100)}%` }}
            />
          </div>
        </section>
      )}

      {leaders.length > 0 && (
        <section className="space-y-4">
          <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
            <div>
              <h2 className="text-2xl font-semibold text-white">Choose Your Leader</h2>
              {selectedSetLabels.length > 0 && (
                <p className="text-xs text-slate-500">Filtered sets: {selectedSetLabels.join(', ')}</p>
              )}
              {(metaOnlyFilter || tournamentOnlyFilter) && (
                <p className="text-xs text-slate-500">
                  {metaOnlyFilter && tournamentOnlyFilter
                    ? 'Meta cards only & Tournament decks only enabled'
                    : metaOnlyFilter
                    ? 'Meta cards only enabled'
                    : 'Tournament decks only enabled'}
                </p>
              )}
            </div>
            <div className="flex items-center gap-3">
              {busy && <span className="text-sm text-slate-400">Working on it...</span>}
              <button
                type="button"
                onClick={handleRegenerateLeaders}
                disabled={leaderMutation.isPending || !prompt.trim()}
                className="rounded-lg border border-primary-400 px-3 py-2 text-xs font-semibold uppercase tracking-wide text-primary-200 transition hover:bg-primary-500/10 disabled:cursor-not-allowed disabled:border-slate-700 disabled:text-slate-500"
              >
                {leaderMutation.isPending ? 'Generating…' : 'Regenerate Leaders'}
              </button>
            </div>
          </div>
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
            {leaders.map((leader) => (
              <LeaderCard
                key={leader.card.id}
                leader={leader}
                isSelected={leader.card.id === selectedLeader?.card.id}
                onSelect={handleLeaderSelect}
              />
            ))}
          </div>
        </section>
      )}

      {deck && (
        <section className="space-y-6">
          <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
            <div className="space-y-1">
              <h2 className="text-2xl font-semibold text-white">Deck List</h2>
              <p className={`text-sm ${deckTotalStatusClass}`}>
                {deck.summary} • {totalQuantity}/50 cards (Leader separate)
              </p>
              {totalQuantity !== 50 && <p className="text-xs text-slate-500">{deckTotalHelperText}</p>}
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <div className="flex items-center gap-3 rounded-lg border border-slate-700 bg-slate-900/50 px-3 py-2">
                <label htmlFor="cards-per-row" className="text-xs font-medium text-slate-300 whitespace-nowrap">
                  Cards per row:
                </label>
                <div className="flex items-center gap-2">
                  <input
                    type="range"
                    id="cards-per-row"
                    min="3"
                    max="6"
                    value={cardsPerRow}
                    onChange={(e) => {
                      const value = parseInt(e.target.value, 10);
                      setCardsPerRow(value);
                      localStorage.setItem('deck-cards-per-row', value.toString());
                    }}
                    className="h-2 w-24 cursor-pointer appearance-none rounded-lg bg-slate-700 accent-primary-500"
                  />
                  <span className="w-6 text-center text-sm font-semibold text-primary-300">{cardsPerRow}</span>
                </div>
              </div>
              <button
                onClick={handleDownload}
                className="inline-flex items-center gap-2 rounded-xl border border-primary-500 px-4 py-2 text-sm font-semibold text-primary-300 transition hover:bg-primary-500/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-200"
              >
                Download Images Zip
              </button>
              <button
                onClick={handleDownloadList}
                className="inline-flex items-center gap-2 rounded-xl border border-slate-700 px-4 py-2 text-sm font-semibold text-slate-200 transition hover:border-primary-400 hover:text-primary-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-200"
              >
                Download Deck List
              </button>
            </div>
          </div>
          <div
            className={`grid gap-5 md:grid-cols-2 ${
              cardsPerRow === 3
                ? 'lg:grid-cols-3'
                : cardsPerRow === 4
                  ? 'lg:grid-cols-4'
                  : cardsPerRow === 5
                    ? 'lg:grid-cols-5'
                    : 'lg:grid-cols-6'
            }`}
          >
            {deck.cards.map((deckCard, index) => (
              <DeckCard
                key={`${deckCard.card.id}-${index}`}
                deckCard={deckCard}
                onSelect={setSelectedDeckCard}
                onIncrement={handleIncrementCard}
                onDecrement={handleDecrementCard}
                onReplace={setReplacementTarget}
                disableIncrement={deckCard.quantity >= 4 || totalQuantity >= 50}
                disableDecrement={deckCard.quantity <= 0}
              />
            ))}
          </div>
          <div className="grid gap-5 md:grid-cols-2">
            <div className="space-y-3 rounded-2xl border border-slate-800 bg-slate-950/70 p-5 shadow-inner shadow-slate-900/40">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-white">Deck Review</h3>
                <button
                  type="button"
                  onClick={handleRerunDeckReview}
                  disabled={totalQuantity !== 50 || reviewLoading}
                  className="rounded-lg border border-primary-400 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-primary-200 transition hover:bg-primary-500/10 disabled:cursor-not-allowed disabled:border-slate-700 disabled:text-slate-500"
                >
                  {reviewLoading ? 'Re-running…' : 'Re-run Deck Review'}
                </button>
              </div>
              <div className="rounded-lg bg-slate-800/50 p-6">
                {deck.deckReview ? (
                  <FormattedReview text={deck.deckReview} />
                ) : totalQuantity === 50 ? (
                  <p className="text-sm text-slate-300">Run the deck review to generate an updated analysis for this adjusted list.</p>
                ) : (
                  <p className="text-sm text-slate-500">Add or remove cards until the deck reaches 50 to enable the review button.</p>
                )}
              </div>
            </div>
            {deckCharacteristics && (
              <div className="space-y-3 rounded-2xl border border-slate-800 bg-slate-950/70 p-5 shadow-inner shadow-slate-900/40">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-white">Deck Characteristics</h3>
                  {deck && (
                    <div className="rounded-lg border border-primary-500/50 bg-primary-500/10 px-4 py-2">
                      <div className="text-xs uppercase tracking-wide text-primary-300">Bounty Score</div>
                      <div className="text-lg font-bold text-primary-200">
                        {computeBountyScore(deck.cards).toLocaleString()}
                      </div>
                    </div>
                  )}
                </div>
                <DeckRadarChart characteristics={deckCharacteristics} />
              </div>
            )}
          </div>
        </section>
      )}

      {selectedDeckCard && (
        <CardDetailModal deckCard={selectedDeckCard} onClose={() => setSelectedDeckCard(null)} />
      )}
      {replacementTarget && deck && (
        <CardReplacementModal
          targetCard={replacementTarget}
          onClose={() => setReplacementTarget(null)}
          leaderCardId={deck.leader.id}
          onReplace={handleReplaceCard}
          excludeCardIds={replacementExcludeIds}
          characteristicOptions={CHARACTERISTIC_CONFIGS.map(({ id, label, description }) => ({
            id,
            label,
            description,
          }))}
          setIds={selectedSetIds}
          metaOnly={metaOnlyFilter}
        />
      )}
    </main>
  );
}
