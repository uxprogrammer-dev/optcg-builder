import { Injectable, Logger } from '@nestjs/common';
import { OptcgCard } from '@shared/types/optcg-card';
import { readFile, readdir, stat } from 'fs/promises';
import { join } from 'path';

import { CardDataRepository } from '../optcg/repositories/card-data.repository';

type TournamentDeckJson = {
  deck_name?: string;
  deckName?: string;
  author?: string;
  date?: string;
  country?: string;
  tournament?: string;
  placement?: string;
  host?: string;
  decklist?: Record<string, number | string>;
};

type TournamentDeckRecord = {
  id: string;
  sourceFile: string;
  deckName: string;
  normalizedName: string;
  leaderId: string;
  leaderName: string;
  cards: string[];
  cardCounts: Map<string, number>;
  keywords: Set<string>;
  metadata: Record<string, unknown>;
  placementScore: number;
};

export type TournamentDeckSummary = {
  deckName: string;
  leaderId: string;
  leaderName: string;
  score: number;
  cards: { cardId: string; quantity: number }[];
  metadata: Record<string, unknown>;
};

@Injectable()
export class TournamentSynergyService {
  private readonly logger = new Logger(TournamentSynergyService.name);

  private readonly decks: TournamentDeckRecord[] = [];
  private readonly deckById = new Map<string, TournamentDeckRecord>();
  private readonly leaderIndex = new Map<string, Set<string>>();
  private readonly archetypeIndex = new Map<string, Set<string>>();
  private readonly keywordIndex = new Map<string, Set<string>>();
  private readonly cardCooccurrence = new Map<string, Map<string, number>>();
  private readonly leaderCardFrequency = new Map<string, Map<string, number>>();
  private readonly leaderCardCooccurrence = new Map<string, Map<string, Map<string, number>>>();
  private readonly stopwords = new Set([
    'the',
    'of',
    'and',
    'or',
    'in',
    'for',
    'deck',
    'with',
    'that',
    'place',
    'tournament',
    'championship',
    'regional',
    'cup',
    'bandai',
    'games',
    'event',
    'team',
    'national',
    'champion',
  ]);

  private loadingPromise: Promise<void> | null = null;

  constructor(private readonly cardRepository: CardDataRepository) {}

  async getCardSynergies(cardId: string, leaderId?: string, limit = 10): Promise<string[]> {
    await this.ensureLoaded();
    const normalizedCard = cardId.trim().toUpperCase();

    let cooccurrence: Map<string, number> | undefined;
    if (leaderId) {
      const leaderMap = this.leaderCardCooccurrence.get(leaderId.trim().toUpperCase());
      cooccurrence = leaderMap?.get(normalizedCard);
    }
    if (!cooccurrence) {
      cooccurrence = this.cardCooccurrence.get(normalizedCard);
    }
    if (!cooccurrence) {
      return [];
    }

    return Array.from(cooccurrence.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, limit)
      .map(([otherCardId]) => otherCardId);
  }

  async getLeaderCardFrequency(leaderId: string, limit = 50): Promise<Record<string, number>> {
    await this.ensureLoaded();
    const normalized = leaderId.trim().toUpperCase();
    const frequency = this.leaderCardFrequency.get(normalized);
    if (!frequency) {
      return {};
    }
    const sorted = Array.from(frequency.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, limit);
    return Object.fromEntries(sorted);
  }

  async isTournamentCard(cardId: string): Promise<boolean> {
    await this.ensureLoaded();
    const normalized = cardId.trim().toUpperCase();
    const baseId = this.baseCardId(normalized);
    
    // First, check if it's a leader in tournament decks (fast path using leaderIndex)
    const leaderDecks = this.leaderIndex.get(normalized);
    if (leaderDecks && leaderDecks.size > 0) {
      return true;
    }
    // Check base ID for leaders - need to check all leader keys
    for (const [leaderKey, deckIds] of this.leaderIndex.entries()) {
      const leaderBase = this.baseCardId(leaderKey);
      if ((leaderKey === normalized || leaderKey === baseId || leaderBase === baseId || leaderBase === normalized) && deckIds.size > 0) {
        return true;
      }
    }
    
    // Also check by iterating through decks directly (for leaders)
    for (const deck of this.decks) {
      // Check if it's the leader of this deck
      if (deck.leaderId === normalized || deck.leaderId === baseId) {
        return true;
      }
      // Check if leader base ID matches
      const deckLeaderBase = this.baseCardId(deck.leaderId);
      if (deckLeaderBase === baseId || deckLeaderBase === normalized) {
        return true;
      }
    }
    
    // Check if card appears in main deck of any tournament deck
    for (const deck of this.decks) {
      // Check if card is in main deck (exact match)
      if (deck.cardCounts.has(normalized)) {
        return true;
      }
      // Check base ID match (for alternate arts)
      if (deck.cardCounts.has(baseId)) {
        return true;
      }
      // Check if any card in deck matches base ID
      for (const deckCardId of deck.cardCounts.keys()) {
        if (this.baseCardId(deckCardId) === baseId) {
          return true;
        }
      }
    }
    
    // Also check leader frequency maps (for main deck cards)
    for (const frequency of this.leaderCardFrequency.values()) {
      if (frequency.has(normalized) || frequency.has(baseId)) {
        return true;
      }
    }
    
    return false;
  }

  async getArchetypeCards(archetype: string, limit = 30): Promise<string[]> {
    await this.ensureLoaded();
    const normalized = this.normalize(archetype);
    if (!normalized.length) {
      return [];
    }

    const deckIds = this.archetypeIndex.get(normalized);
    if (!deckIds || deckIds.size === 0) {
      return [];
    }

    const counts = new Map<string, number>();
    for (const deckId of deckIds) {
      const deck = this.deckById.get(deckId);
      if (!deck) {
        continue;
      }
      for (const [cardId, quantity] of deck.cardCounts.entries()) {
        counts.set(cardId, (counts.get(cardId) ?? 0) + quantity);
      }
    }

    return Array.from(counts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, limit)
      .map(([cardId]) => cardId);
  }

  async findSimilarTournamentDecks(
    leaderId: string,
    prompt?: string,
    archetype?: string,
    limit = 5,
  ): Promise<TournamentDeckSummary[]> {
    await this.ensureLoaded();

    const normalizedLeader = leaderId.trim().toUpperCase();
    const baseLeader = this.baseCardId(normalizedLeader);
    const promptTokens = this.tokenize(prompt ?? '');
    const archetypeTokens = this.tokenize(archetype ?? '');

    const candidateDeckIds = new Set<string>();

    const leaderDecks = this.leaderIndex.get(normalizedLeader);
    leaderDecks?.forEach((id) => candidateDeckIds.add(id));

    if (candidateDeckIds.size < limit) {
      for (const [leaderKey, decks] of this.leaderIndex.entries()) {
        if (leaderKey === normalizedLeader) {
          continue;
        }
        if (this.baseCardId(leaderKey) === baseLeader) {
          decks.forEach((id) => candidateDeckIds.add(id));
        }
      }
    }

    if (archetypeTokens.length) {
      for (const token of archetypeTokens) {
        const keywordDecks = this.keywordIndex.get(token);
        keywordDecks?.forEach((id) => candidateDeckIds.add(id));
      }
    }

    if (promptTokens.length) {
      for (const token of promptTokens) {
        const keywordDecks = this.keywordIndex.get(token);
        keywordDecks?.forEach((id) => candidateDeckIds.add(id));
      }
    }

    if (candidateDeckIds.size === 0) {
      this.decks.forEach((deck) => candidateDeckIds.add(deck.id));
    }

    const scored: Array<{ deck: TournamentDeckRecord; score: number }> = [];
    for (const deckId of candidateDeckIds) {
      const deck = this.deckById.get(deckId);
      if (!deck) {
        continue;
      }
      let score = deck.placementScore;
      if (deck.leaderId === normalizedLeader) {
        score += 40;
      } else if (this.baseCardId(deck.leaderId) === baseLeader) {
        score += 20;
      }

      const matchingTokens = promptTokens.filter((token) => deck.keywords.has(token));
      score += matchingTokens.length * 3;

      const matchingArchetype = archetypeTokens.filter((token) => deck.keywords.has(token));
      score += matchingArchetype.length * 4;

      if (deck.normalizedName.length && archetype) {
        if (deck.normalizedName.includes(this.normalize(archetype))) {
          score += 6;
        }
      }

      if (score > 0) {
        scored.push({ deck, score });
      }
    }

    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, limit).map(({ deck, score }) => this.toSummary(deck, score));
  }

  private async ensureLoaded(): Promise<void> {
    if (!this.loadingPromise) {
      this.loadingPromise = this.load().catch((error) => {
        this.logger.error(`Failed to load tournament decks: ${(error as Error).message}`);
        this.loadingPromise = null;
        throw error;
      });
    }
    await this.loadingPromise;
  }

  private async load(): Promise<void> {
    const directory = join(__dirname, '../../../data/tournaments');
    try {
      const stats = await stat(directory);
      if (!stats.isDirectory()) {
        this.logger.warn(`Tournament directory is not a folder: ${directory}`);
        return;
      }
    } catch (error) {
      this.logger.warn(`Tournament directory not found: ${directory}`);
      return;
    }

    const files = await readdir(directory);
    const jsonFiles = files.filter((name) => name.toLowerCase().endsWith('.json'));
    if (!jsonFiles.length) {
      this.logger.warn(`No tournament deck files found in ${directory}`);
      return;
    }

    for (const fileName of jsonFiles) {
      const path = join(directory, fileName);
      try {
        const raw = await readFile(path, 'utf8');
        const payload = JSON.parse(raw) as unknown;
        if (!Array.isArray(payload)) {
          this.logger.warn(`Tournament file ${fileName} does not contain an array of decks.`);
          continue;
        }

        for (let index = 0; index < payload.length; index += 1) {
          const entry = payload[index];
          if (!entry || typeof entry !== 'object') {
            continue;
          }
          const deck = await this.normaliseDeck(entry as TournamentDeckJson, fileName, index);
          if (!deck) {
            continue;
          }

          this.decks.push(deck);
          this.deckById.set(deck.id, deck);
          this.indexDeck(deck);
        }
      } catch (error) {
        this.logger.warn(`Failed to process ${fileName}: ${(error as Error).message}`);
      }
    }

    this.logger.log(`Loaded ${this.decks.length} tournament decks for synergy knowledge.`);
  }

  private async normaliseDeck(
    entry: TournamentDeckJson,
    sourceFile: string,
    index: number,
  ): Promise<TournamentDeckRecord | null> {
    const deckName = entry.deck_name ?? entry.deckName ?? 'Tournament Deck';
    const decklist = entry.decklist;
    if (!decklist || typeof decklist !== 'object') {
      this.logger.debug(`Skipping deck ${deckName} from ${sourceFile}: missing decklist.`);
      return null;
    }

    let leaderId: string | null = null;
    let leaderName = '';
    const cardCounts = new Map<string, number>();

    for (const [rawCardId, rawQuantity] of Object.entries(decklist)) {
      if (!rawCardId) {
        continue;
      }

      const quantity = this.parseQuantity(rawQuantity);
      if (!Number.isFinite(quantity) || quantity <= 0) {
        continue;
      }

      const card = await this.resolveCard(rawCardId);
      if (!card) {
        this.logger.debug(
          `Skipping unknown card ${rawCardId} in deck ${deckName} from ${sourceFile}.`,
        );
        continue;
      }

      const resolvedId = card.id.toUpperCase();
      if ((card.type ?? '').toLowerCase() === 'leader') {
        leaderId = resolvedId;
        leaderName = card.name ?? resolvedId;
        continue;
      }

      cardCounts.set(resolvedId, (cardCounts.get(resolvedId) ?? 0) + quantity);
    }

    if (!leaderId) {
      this.logger.debug(`Skipping deck ${deckName} from ${sourceFile}: leader not identified.`);
      return null;
    }

    const mainDeck: string[] = [];
    for (const [cardId, quantity] of cardCounts.entries()) {
      for (let i = 0; i < quantity; i += 1) {
        mainDeck.push(cardId);
      }
    }

    if (mainDeck.length < 40) {
      this.logger.debug(
        `Skipping deck ${deckName} from ${sourceFile}: insufficient cards (${mainDeck.length}).`,
      );
      return null;
    }

    const metadata: Record<string, unknown> = {
      source: 'tournament',
      sourceFile,
      deckName,
      author: entry.author ?? null,
      date: entry.date ?? null,
      country: entry.country ?? null,
      tournament: entry.tournament ?? null,
      placement: entry.placement ?? null,
      host: entry.host ?? null,
    };

    const keywords = new Set<string>();
    this.tokenize(deckName).forEach((token) => keywords.add(token));
    this.tokenize(leaderName).forEach((token) => keywords.add(token));
    if (entry.tournament) {
      this.tokenize(entry.tournament).forEach((token) => keywords.add(token));
    }
    if (entry.placement) {
      this.tokenize(entry.placement).forEach((token) => keywords.add(token));
    }
    if (entry.country) {
      this.tokenize(entry.country).forEach((token) => keywords.add(token));
    }

    const record: TournamentDeckRecord = {
      id: `${sourceFile}#${index}`,
      sourceFile,
      deckName,
      normalizedName: this.normalize(deckName),
      leaderId,
      leaderName,
      cards: mainDeck,
      cardCounts,
      keywords,
      metadata,
      placementScore: this.placementScore(entry.placement ?? ''),
    };

    return record;
  }

  private indexDeck(deck: TournamentDeckRecord): void {
    const leaderDecks = this.leaderIndex.get(deck.leaderId) ?? new Set<string>();
    leaderDecks.add(deck.id);
    this.leaderIndex.set(deck.leaderId, leaderDecks);

    if (deck.normalizedName.length) {
      const archetypeDecks = this.archetypeIndex.get(deck.normalizedName) ?? new Set<string>();
      archetypeDecks.add(deck.id);
      this.archetypeIndex.set(deck.normalizedName, archetypeDecks);
    }

    for (const keyword of deck.keywords) {
      if (!keyword.length) {
        continue;
      }
      const decks = this.keywordIndex.get(keyword) ?? new Set<string>();
      decks.add(deck.id);
      this.keywordIndex.set(keyword, decks);
    }

    this.addCooccurrence(deck);
  }

  private addCooccurrence(deck: TournamentDeckRecord): void {
    const uniqueCards = Array.from(deck.cardCounts.keys());

    const leaderFrequency = this.leaderCardFrequency.get(deck.leaderId) ?? new Map<string, number>();
    for (const [cardId, quantity] of deck.cardCounts.entries()) {
      leaderFrequency.set(cardId, (leaderFrequency.get(cardId) ?? 0) + quantity);
    }
    this.leaderCardFrequency.set(deck.leaderId, leaderFrequency);

    const leaderSpecific =
      this.leaderCardCooccurrence.get(deck.leaderId) ??
      new Map<string, Map<string, number>>();

    for (const card of uniqueCards) {
      const globalMap = this.cardCooccurrence.get(card) ?? new Map<string, number>();
      for (const other of uniqueCards) {
        if (other === card) {
          continue;
        }
        globalMap.set(other, (globalMap.get(other) ?? 0) + 1);

        const leaderMap = leaderSpecific.get(card) ?? new Map<string, number>();
        leaderMap.set(other, (leaderMap.get(other) ?? 0) + 1);
        leaderSpecific.set(card, leaderMap);
      }
      this.cardCooccurrence.set(card, globalMap);
    }

    this.leaderCardCooccurrence.set(deck.leaderId, leaderSpecific);
  }

  private async resolveCard(cardId: string): Promise<OptcgCard | null> {
    const normalized = cardId.trim().toUpperCase();
    let card = await this.cardRepository.findById(normalized);
    if (card) {
      return card;
    }

    const baseId = this.baseCardId(normalized);
    if (baseId !== normalized) {
      card = await this.cardRepository.findById(baseId);
      if (card) {
        return card;
      }
    }

    return null;
  }

  private normalize(value: string): string {
    return value
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, ' ')
      .trim()
      .replace(/\s+/g, ' ');
  }

  private tokenize(value: string): string[] {
    if (!value) {
      return [];
    }
    return this.normalize(value)
      .split(' ')
      .filter((token) => token.length && !this.stopwords.has(token));
  }

  private baseCardId(cardId: string): string {
    const normalized = cardId.trim().toUpperCase();
    const [base] = normalized.split('_');
    return base;
  }

  private parseQuantity(value: number | string): number {
    if (typeof value === 'number') {
      return Number.isFinite(value) ? value : 0;
    }
    const parsed = parseInt(value, 10);
    return Number.isFinite(parsed) ? parsed : 0;
  }

  private placementScore(placement: string): number {
    if (!placement) {
      return 1;
    }
    const lower = placement.toLowerCase();
    if (lower.includes('1st') || lower.includes('champion') || lower.includes('winner')) {
      return 15;
    }
    if (lower.includes('2nd') || lower.includes('3rd') || lower.includes('finalist')) {
      return 8;
    }
    if (
      lower.includes('t4') ||
      lower.includes('top4') ||
      lower.includes('top 4') ||
      lower.includes('t8') ||
      lower.includes('top8') ||
      lower.includes('top 8') ||
      lower.includes('t16') ||
      lower.includes('top16') ||
      lower.includes('top 16')
    ) {
      return 5;
    }
    return 2;
  }

  private toSummary(deck: TournamentDeckRecord, score: number): TournamentDeckSummary {
    return {
      deckName: deck.deckName,
      leaderId: deck.leaderId,
      leaderName: deck.leaderName,
      score,
      cards: Array.from(deck.cardCounts.entries())
        .map(([cardId, quantity]) => ({ cardId, quantity }))
        .sort((a, b) => b.quantity - a.quantity),
      metadata: { ...deck.metadata },
    };
  }
}


