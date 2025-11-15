import { Injectable, Logger } from '@nestjs/common';
import { CardDataRepository } from '../optcg/repositories/card-data.repository';
import { TournamentSynergyService } from './tournament-synergy.service';
import { OptcgCard } from '@shared/types/optcg-card';
import metaCardConfig from '../../data/meta/meta-cards.json';
import bannedCardConfig from '../../data/meta/banned-cards.json';
import setsIndex from '../../data/sets/en/index.json';

export type CardKnowledgeMetadata = {
  id: string;
  name: string;
  type: string;
  colors: string[];
  subtypes: string[];
  attribute: string | null;
  setName: string | null;
  cost: string | null;
  power: string | null;
  life: string | null;
  rarity: string | null;
  counter: number;
  text: string | null;
  keywords: string[];
  isVanilla: boolean;
};

type CardKnowledgeRecord = {
  id: string;
  name: string;
  type: string;
  colors: string[];
  subtypes: string[];
  attribute: string | null;
  setName: string | null;
  cost: string | null;
  power: string | null;
  life: string | null;
  rarity: string | null;
  counter: number;
  text: string | null;
  keywordsSet: Set<string>;
  isVanilla: boolean;
};

@Injectable()
export class CardKnowledgeService {
  private readonly logger = new Logger(CardKnowledgeService.name);
  private readonly cards = new Map<string, CardKnowledgeRecord>();
  private readonly keywordIndex = new Map<string, Set<string>>();
  private readonly colorIndex = new Map<string, Set<string>>();
  private readonly typeIndex = new Map<string, Set<string>>();
  private readonly metaCardIds = new Set<string>((metaCardConfig.metaCardIds ?? []).map((id) => id.toUpperCase()));
  private readonly bannedCardIds = new Set<string>((bannedCardConfig.bannedCardIds ?? []).map((id) => id.toUpperCase()));
  private loadingPromise: Promise<void> | null = null;
  private readonly stopwords = new Set([
    'the',
    'of',
    'and',
    'or',
    'in',
    'your',
    'this',
    'that',
    'from',
    'with',
    'you',
    'for',
    'turn',
    'once',
    'per',
    'all',
    'any',
    'gain',
    'during',
    'when',
    'cost',
    'life',
    'power',
    'card',
    'cards',
    'don',
    '1000',
    '5000',
    'activate',
    'main',
    'opponent',
    'character',
    'characters',
    'battle',
    'phase',
    'deck',
    'search',
    'added',
  ]);

  constructor(
    private readonly cardDataRepository: CardDataRepository,
    private readonly tournamentSynergy: TournamentSynergyService,
  ) {}

  async listAllMetadata(): Promise<CardKnowledgeMetadata[]> {
    await this.ensureLoaded();
    return Array.from(this.cards.values()).map((card) => this.toMetadata(card));
  }

  async listAvailableSets(): Promise<{ id: string; label: string }[]> {
    await this.ensureLoaded();
    
    // Start with sets from index.json as the primary source
    // Map key: normalized set code (e.g., "OP13"), value: { originalId, label }
    const setMap = new Map<string, { originalId: string; label: string }>();
    
    // Load sets from index.json
    if (Array.isArray(setsIndex)) {
      for (const set of setsIndex) {
        if (set.id && set.name) {
          const normalized = this.normalizeSetCode(set.id);
          setMap.set(normalized, { originalId: set.id, label: set.name });
        }
      }
    }
    
    // Merge with sets found in cards (for any sets not in index.json)
    for (const card of this.cards.values()) {
      const setCode = this.getSetCode(card.id);
      if (!setCode) {
        continue;
      }
      const normalized = this.normalizeSetCode(setCode);
      if (!setMap.has(normalized)) {
        const label = card.setName ?? setCode;
        setMap.set(normalized, { originalId: setCode, label: label.replace(/^[\s-]+|\s+$/g, '') });
      }
    }
    
    // Return sorted list, using original IDs from index.json when available
    return Array.from(setMap.entries())
      .map(([normalized, { originalId, label }]) => ({
        id: originalId,
        label,
      }))
      .sort((a, b) => {
        // Sort by normalized code for consistent ordering
        const normA = this.normalizeSetCode(a.id);
        const normB = this.normalizeSetCode(b.id);
        return normA.localeCompare(normB);
      });
  }

  async getCardMetadata(cardId: string): Promise<CardKnowledgeMetadata | null> {
    await this.ensureLoaded();
    const card = this.cards.get(cardId.toUpperCase());
    return card ? this.toMetadata(card) : null;
  }

  async findByKeyword(keyword: string): Promise<CardKnowledgeMetadata[]> {
    await this.ensureLoaded();
    const normalized = this.normalize(keyword);
    const ids = this.keywordIndex.get(normalized);
    if (!ids) {
      return [];
    }
    return Array.from(ids)
      .map((id) => this.cards.get(id))
      .filter((card): card is CardKnowledgeRecord => Boolean(card))
      .map((card) => this.toMetadata(card));
  }

  async findByKeywords(keywords: string[]): Promise<CardKnowledgeMetadata[]> {
    await this.ensureLoaded();
    const results = new Set<CardKnowledgeRecord>();
    keywords.forEach((keyword) => {
      const normalized = this.normalize(keyword);
      const ids = this.keywordIndex.get(normalized);
      if (!ids) {
        return;
      }
      ids.forEach((id) => {
        const card = this.cards.get(id);
        if (card) {
          results.add(card);
        }
      });
    });
    return Array.from(results).map((card) => this.toMetadata(card));
  }

  async getCardsByColor(color: string): Promise<CardKnowledgeMetadata[]> {
    await this.ensureLoaded();
    const normalized = color.trim().toLowerCase();
    const ids = this.colorIndex.get(normalized);
    if (!ids) {
      return [];
    }
    return Array.from(ids)
      .map((id) => this.cards.get(id))
      .filter((card): card is CardKnowledgeRecord => Boolean(card))
      .map((card) => this.toMetadata(card));
  }

  async getCardsByType(type: string): Promise<CardKnowledgeMetadata[]> {
    await this.ensureLoaded();
    const normalized = type.trim().toLowerCase();
    const ids = this.typeIndex.get(normalized);
    if (!ids) {
      return [];
    }
    return Array.from(ids)
      .map((id) => this.cards.get(id))
      .filter((card): card is CardKnowledgeRecord => Boolean(card))
      .map((card) => this.toMetadata(card));
  }

  isMetaCard(cardId: string): boolean {
    return this.metaCardIds.has(cardId.toUpperCase());
  }

  isBannedCard(cardId: string): boolean {
    return this.bannedCardIds.has(cardId.toUpperCase());
  }

  async isTournamentCard(cardId: string): Promise<boolean> {
    return this.tournamentSynergy.isTournamentCard(cardId);
  }

  private getSetCode(cardId: string): string {
    const match = cardId.toUpperCase().match(/^([A-Z0-9]+)-/);
    return match ? match[1] : '';
  }

  /**
   * Normalizes set codes for matching (e.g., "OP-13" → "OP13", "OP13" → "OP13").
   * This allows matching between formats with and without dashes.
   */
  private normalizeSetCode(setCode: string): string {
    // Remove dashes and convert to uppercase for consistent matching
    // Examples: "OP-13" → "OP13", "ST-01" → "ST01", "EB-01" → "EB01"
    return setCode.toUpperCase().replace(/-/g, '');
  }

  private toMetadata(card: CardKnowledgeRecord): CardKnowledgeMetadata {
    return {
      id: card.id,
      name: card.name,
      type: card.type,
      colors: [...card.colors],
      subtypes: [...card.subtypes],
      attribute: card.attribute,
      setName: card.setName,
      cost: card.cost,
      power: card.power,
      life: card.life,
      rarity: card.rarity,
      counter: card.counter,
      text: card.text,
      keywords: Array.from(card.keywordsSet),
      isVanilla: card.isVanilla,
    };
  }

  private async ensureLoaded(): Promise<void> {
    if (this.cards.size > 0) {
      return;
    }
    if (!this.loadingPromise) {
      this.loadingPromise = this.load();
    }
    await this.loadingPromise;
  }

  private async load(): Promise<void> {
    try {
      const cards = await this.cardDataRepository.getAllCards();
      cards.forEach((card) => this.indexCard(card));
      this.logger.log(
        `Indexed ${this.cards.size} cards with ${this.keywordIndex.size} keywords for local intent matching`,
      );
    } catch (error) {
      this.logger.error(`Failed to load card knowledge: ${(error as Error).message}`);
    }
  }

  private indexCard(card: OptcgCard): void {
    const id = card.id.toUpperCase();
    const keywords = new Set<string>();
    const colors = this.splitValues(card.color);
    const subtypes = card.subtypes.map((value) => value.trim()).filter(Boolean);
    const attribute = card.attribute?.trim() || null;
    const setName = card.setName?.trim() || null;
    const text = this.normalizeText(card.text);
    const isVanilla = !text;
    const type = card.type.trim();
    const rarity = card.rarity ?? null;
    const cost = card.cost ?? null;
    const power = card.power ?? null;
    const life = card.life ?? null;
    const counter = card.raw?.counter_amount ?? 0;

    this.addKeywordsFromValue(card.name, id, keywords);
    this.tokenize(card.name).forEach((token) => this.addKeywordsFromToken(token, id, keywords));

    subtypes.forEach((subtype) => {
      this.addKeywordsFromValue(subtype, id, keywords);
      this.tokenize(subtype).forEach((token) => this.addKeywordsFromToken(token, id, keywords));
    });

    colors.forEach((color) => {
      this.addKeywordsFromToken(color, id, keywords);
      this.indexColor(color, id);
    });

    if (attribute) {
      this.addKeywordsFromValue(attribute, id, keywords);
    }

    if (setName) {
      this.addKeywordsFromValue(setName, id, keywords);
      this.tokenize(setName).forEach((token) => this.addKeywordsFromToken(token, id, keywords));
    }

    if (text) {
      this.tokenize(text).forEach((token) => this.addKeywordsFromToken(token, id, keywords));
    }

    this.indexType(type, id);

    this.cards.set(id, {
      id,
      name: card.name,
      type,
      colors,
      subtypes,
      attribute,
      setName,
      cost,
      power,
      life,
      rarity,
      counter,
      text,
      keywordsSet: keywords,
      isVanilla,
    });
  }

  private addKeywordsFromValue(value: string, cardId: string, keywords: Set<string>): void {
    const normalized = this.normalize(value);
    if (!normalized) {
      return;
    }
    this.addKeywordsFromToken(normalized, cardId, keywords);
  }

  private addKeywordsFromToken(token: string, cardId: string, keywords: Set<string>): void {
    const normalized = this.normalize(token);
    if (!normalized || normalized.length < 2 || this.stopwords.has(normalized)) {
      return;
    }
    keywords.add(normalized);
    const bucket = this.keywordIndex.get(normalized) ?? new Set<string>();
    bucket.add(cardId);
    this.keywordIndex.set(normalized, bucket);
  }

  private indexColor(color: string, cardId: string): void {
    const normalized = color.trim().toLowerCase();
    if (!normalized) {
      return;
    }
    const bucket = this.colorIndex.get(normalized) ?? new Set<string>();
    bucket.add(cardId);
    this.colorIndex.set(normalized, bucket);
  }

  private indexType(type: string, cardId: string): void {
    const normalized = type.trim().toLowerCase();
    if (!normalized) {
      return;
    }
    const bucket = this.typeIndex.get(normalized) ?? new Set<string>();
    bucket.add(cardId);
    this.typeIndex.set(normalized, bucket);
  }

  private splitValues(value: string | null): string[] {
    if (!value) {
      return [];
    }
    return value
      .split(/[\/,]/)
      .map((part) => this.normalize(part))
      .filter((part) => !!part && !this.stopwords.has(part));
  }

  private normalize(value: string): string {
    return value.toLowerCase().replace(/[^a-z0-9\s]/g, ' ').replace(/\s+/g, ' ').trim();
  }

  private tokenize(value: string): string[] {
    return this.normalize(value)
      .split(' ')
      .filter((part) => part.length >= 2 && !this.stopwords.has(part));
  }

  private normalizeText(text: string | null): string | null {
    if (!text) {
      return null;
    }
    const trimmed = text.trim();
    if (!trimmed || trimmed.toLowerCase() === 'null') {
      return null;
    }
    return trimmed;
  }
}
