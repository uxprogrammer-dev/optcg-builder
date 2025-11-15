import { Injectable, Logger } from '@nestjs/common';
import * as fs from 'fs/promises';
import * as path from 'path';

export type LeaderKeyCard = {
  cardId: string;
  role: string;
  rationale: string;
};

export type LeaderStrategy = {
  leaderId: string;
  leaderName: string;
  archetype: string;
  strategy: string;
  keyCards: LeaderKeyCard[];
  winConditions: string[];
  typicalRatios: {
    characters: number;
    events: number;
    stages: number;
  };
  costCurve: {
    '0-2': number;
    '3-4': number;
    '5-6': number;
    '7+': number;
  };
};

type LeaderStrategiesPayload = {
  version: string;
  strategies: LeaderStrategy[];
};

@Injectable()
export class LeaderStrategyService {
  private readonly logger = new Logger(LeaderStrategyService.name);
  private readonly strategiesPath: string;
  private cache: LeaderStrategy[] | null = null;
  private loadingPromise: Promise<LeaderStrategy[]> | null = null;

  constructor() {
    const dataDir = path.join(process.cwd(), 'data', 'strategies');
    this.strategiesPath = path.join(dataDir, 'leader-strategies.json');
  }

  /**
   * Load leader strategies from disk (cached after first read).
   */
  async loadStrategies(force = false): Promise<LeaderStrategy[]> {
    if (this.cache && !force) {
      return this.cache;
    }

    if (this.loadingPromise && !force) {
      return this.loadingPromise;
    }

    this.loadingPromise = this.readStrategiesFromDisk();
    this.cache = await this.loadingPromise;
    this.loadingPromise = null;
    return this.cache;
  }

  /**
   * Get strategy for a specific leader.
   */
  async getStrategyForLeader(leaderId: string): Promise<LeaderStrategy | null> {
    const strategies = await this.loadStrategies();
    return strategies.find((s) => s.leaderId.toUpperCase() === leaderId.toUpperCase()) || null;
  }

  /**
   * Get key cards for a leader.
   */
  async getKeyCardsForLeader(leaderId: string): Promise<LeaderKeyCard[]> {
    const strategy = await this.getStrategyForLeader(leaderId);
    return strategy?.keyCards || [];
  }

  /**
   * Get strategy description for prompt generation.
   */
  async getStrategyDescription(leaderId: string): Promise<string | null> {
    const strategy = await this.getStrategyForLeader(leaderId);
    if (!strategy) {
      return null;
    }

    const parts: string[] = [];
    parts.push(`Strategy: ${strategy.strategy}`);
    if (strategy.winConditions.length > 0) {
      parts.push(`Win Conditions: ${strategy.winConditions.join(', ')}`);
    }
    if (strategy.keyCards.length > 0) {
      const keyCardNames = strategy.keyCards.map((kc) => kc.cardId).join(', ');
      parts.push(`Key Cards: ${keyCardNames}`);
    }

    return parts.join('. ');
  }

  /**
   * Get typical deck ratios for a leader.
   */
  async getTypicalRatios(leaderId: string): Promise<LeaderStrategy['typicalRatios'] | null> {
    const strategy = await this.getStrategyForLeader(leaderId);
    return strategy?.typicalRatios || null;
  }

  /**
   * Get cost curve for a leader.
   */
  async getCostCurve(leaderId: string): Promise<LeaderStrategy['costCurve'] | null> {
    const strategy = await this.getStrategyForLeader(leaderId);
    return strategy?.costCurve || null;
  }

  private async readStrategiesFromDisk(): Promise<LeaderStrategy[]> {
    try {
      const raw = await fs.readFile(this.strategiesPath, 'utf-8');
      const payload = JSON.parse(raw) as LeaderStrategiesPayload;

      if (!payload.strategies || !Array.isArray(payload.strategies)) {
        this.logger.warn(`Leader strategies file at ${this.strategiesPath} is missing expected fields.`);
        return [];
      }

      this.logger.log(`Loaded ${payload.strategies.length} leader strategy definitions`);
      return payload.strategies;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
        this.logger.warn(`Leader strategies file not found at ${this.strategiesPath}. Continuing without strategy data.`);
        return [];
      }
      this.logger.error(
        `Failed to read leader strategies file at ${this.strategiesPath}: ${(error as Error).message}`,
        error as Error,
      );
      return [];
    }
  }
}

