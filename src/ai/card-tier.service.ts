import { Injectable, Logger } from '@nestjs/common';
import * as fs from 'fs/promises';
import * as path from 'path';

export type CardTier = 'S' | 'A' | 'B' | 'C';

export type CardTierData = {
  tier: CardTier;
  description: string;
  cards: string[];
};

type CardTiersPayload = {
  version: string;
  tiers: Record<CardTier, CardTierData>;
  leaderSpecific?: Record<string, Record<CardTier, string[]>>;
};

@Injectable()
export class CardTierService {
  private readonly logger = new Logger(CardTierService.name);
  private readonly tiersPath: string;
  private cache: CardTiersPayload | null = null;
  private loadingPromise: Promise<CardTiersPayload | null> | null = null;

  constructor() {
    const dataDir = path.join(process.cwd(), 'data', 'meta');
    this.tiersPath = path.join(dataDir, 'card-tiers.json');
  }

  /**
   * Load card tier data from disk (cached after first read).
   */
  async loadTiers(force = false): Promise<CardTiersPayload | null> {
    if (this.cache && !force) {
      return this.cache;
    }

    if (this.loadingPromise && !force) {
      return this.loadingPromise;
    }

    this.loadingPromise = this.readTiersFromDisk();
    this.cache = await this.loadingPromise;
    this.loadingPromise = null;
    return this.cache;
  }

  /**
   * Get tier for a card (checks leader-specific first, then general).
   */
  async getCardTier(cardId: string, leaderId?: string): Promise<CardTier | null> {
    const payload = await this.loadTiers();
    if (!payload) {
      return null;
    }

    const normalizedCardId = cardId.toUpperCase();

    // Check leader-specific tiers first
    if (leaderId && payload.leaderSpecific) {
      const leaderTiers = payload.leaderSpecific[leaderId.toUpperCase()];
      if (leaderTiers) {
        for (const tier of ['S', 'A', 'B', 'C'] as CardTier[]) {
          if (leaderTiers[tier]?.includes(normalizedCardId)) {
            return tier;
          }
        }
      }
    }

    // Check general tiers
    for (const tier of ['S', 'A', 'B', 'C'] as CardTier[]) {
      if (payload.tiers[tier]?.cards.includes(normalizedCardId)) {
        return tier;
      }
    }

    return null;
  }

  /**
   * Get all cards in a specific tier (for a leader if provided).
   */
  async getCardsByTier(tier: CardTier, leaderId?: string): Promise<string[]> {
    const payload = await this.loadTiers();
    if (!payload) {
      return [];
    }

    const cards: string[] = [];

    // Get leader-specific cards first
    if (leaderId && payload.leaderSpecific) {
      const leaderTiers = payload.leaderSpecific[leaderId.toUpperCase()];
      if (leaderTiers?.[tier]) {
        cards.push(...leaderTiers[tier]);
      }
    }

    // Add general tier cards
    if (payload.tiers[tier]?.cards) {
      cards.push(...payload.tiers[tier].cards);
    }

    // Remove duplicates
    return Array.from(new Set(cards));
  }

  /**
   * Get tier score for card scoring (S=4, A=3, B=2, C=1, null=0).
   */
  async getTierScore(cardId: string, leaderId?: string): Promise<number> {
    const tier = await this.getCardTier(cardId, leaderId);
    if (!tier) {
      return 0;
    }

    const tierScores: Record<CardTier, number> = {
      S: 4,
      A: 3,
      B: 2,
      C: 1,
    };

    return tierScores[tier];
  }

  /**
   * Check if a card is in top tiers (S or A).
   */
  async isTopTier(cardId: string, leaderId?: string): Promise<boolean> {
    const tier = await this.getCardTier(cardId, leaderId);
    return tier === 'S' || tier === 'A';
  }

  private async readTiersFromDisk(): Promise<CardTiersPayload | null> {
    try {
      const raw = await fs.readFile(this.tiersPath, 'utf-8');
      const payload = JSON.parse(raw) as CardTiersPayload;

      if (!payload.tiers) {
        this.logger.warn(`Card tiers file at ${this.tiersPath} is missing expected fields.`);
        return null;
      }

      this.logger.log(`Loaded card tier data with ${Object.keys(payload.tiers).length} tiers`);
      return payload;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
        this.logger.warn(`Card tiers file not found at ${this.tiersPath}. Continuing without tier data.`);
        return null;
      }
      this.logger.error(
        `Failed to read card tiers file at ${this.tiersPath}: ${(error as Error).message}`,
        error as Error,
      );
      return null;
    }
  }
}

