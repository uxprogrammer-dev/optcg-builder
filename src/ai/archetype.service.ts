import { Injectable, Logger } from '@nestjs/common';
import * as fs from 'fs/promises';
import * as path from 'path';

export type ArchetypeDefinition = {
  id: string;
  name: string;
  description: string;
  keySubtypes: string[];
  keyColors: string[];
  strategy: string;
  keyCards: string[];
  typicalRatios: {
    characters: number;
    events: number;
    stages: number;
  };
};

type ArchetypesPayload = {
  version: string;
  archetypes: ArchetypeDefinition[];
};

@Injectable()
export class ArchetypeService {
  private readonly logger = new Logger(ArchetypeService.name);
  private readonly archetypesPath: string;
  private cache: ArchetypeDefinition[] | null = null;
  private loadingPromise: Promise<ArchetypeDefinition[]> | null = null;

  constructor() {
    const dataDir = path.join(process.cwd(), 'data', 'archetypes');
    this.archetypesPath = path.join(dataDir, 'archetypes.json');
  }

  /**
   * Load archetype definitions from disk (cached after first read).
   */
  async loadArchetypes(force = false): Promise<ArchetypeDefinition[]> {
    if (this.cache && !force) {
      return this.cache;
    }

    if (this.loadingPromise && !force) {
      return this.loadingPromise;
    }

    this.loadingPromise = this.readArchetypesFromDisk();
    this.cache = await this.loadingPromise;
    this.loadingPromise = null;
    return this.cache;
  }

  /**
   * Get archetype by ID.
   */
  async getArchetypeById(id: string): Promise<ArchetypeDefinition | null> {
    const archetypes = await this.loadArchetypes();
    return archetypes.find((a) => a.id === id) || null;
  }

  /**
   * Find archetypes matching given subtypes and colors.
   */
  async findArchetypesBySubtypesAndColors(
    subtypes: string[],
    colors: string[],
  ): Promise<ArchetypeDefinition[]> {
    const archetypes = await this.loadArchetypes();
    const normalizedSubtypes = new Set(subtypes.map((s) => s.toLowerCase().trim()));
    const normalizedColors = new Set(colors.map((c) => c.toLowerCase().trim()));

    return archetypes.filter((archetype) => {
      const hasMatchingSubtype = archetype.keySubtypes.some((subtype) =>
        normalizedSubtypes.has(subtype.toLowerCase().trim()),
      );
      const hasMatchingColor = archetype.keyColors.some((color) =>
        normalizedColors.has(color.toLowerCase().trim()),
      );
      return hasMatchingSubtype || hasMatchingColor;
    });
  }

  /**
   * Get archetype description for prompt generation.
   */
  async getArchetypeDescription(archetypeId: string): Promise<string | null> {
    const archetype = await this.getArchetypeById(archetypeId);
    if (!archetype) {
      return null;
    }
    return `${archetype.name}: ${archetype.description}. Strategy: ${archetype.strategy}`;
  }

  private async readArchetypesFromDisk(): Promise<ArchetypeDefinition[]> {
    try {
      const raw = await fs.readFile(this.archetypesPath, 'utf-8');
      const payload = JSON.parse(raw) as ArchetypesPayload;

      if (!payload.archetypes || !Array.isArray(payload.archetypes)) {
        this.logger.warn(`Archetypes file at ${this.archetypesPath} is missing expected fields.`);
        return [];
      }

      this.logger.log(`Loaded ${payload.archetypes.length} archetype definitions`);
      return payload.archetypes;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
        this.logger.warn(`Archetypes file not found at ${this.archetypesPath}. Continuing without archetype data.`);
        return [];
      }
      this.logger.error(
        `Failed to read archetypes file at ${this.archetypesPath}: ${(error as Error).message}`,
        error as Error,
      );
      return [];
    }
  }
}

