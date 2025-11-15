import { Injectable, Logger } from '@nestjs/common';
import { readFileSync } from 'fs';
import { join } from 'path';

type LeaderIndexEntry = {
  id: string;
  name: string;
  colors: string[];
  subtypes: string[];
  setName?: string | null;
};

export type LeaderKnowledgeMetadata = {
  id: string;
  name: string;
  colors: string[];
  subtypes: string[];
  setName: string | null;
  keywords: string[];
};

type LeaderKnowledgeRecord = {
  id: string;
  name: string;
  colors: string[];
  subtypes: string[];
  setName: string | null;
  keywords: Set<string>;
};

@Injectable()
export class LeaderKnowledgeService {
  private readonly logger = new Logger(LeaderKnowledgeService.name);
  private readonly leaders = new Map<string, LeaderKnowledgeRecord>();
  private readonly colorIndex = new Map<string, Set<string>>();
  private readonly keywordToLeaderIds = new Map<string, Set<string>>();
  private readonly nameToLeaderIds = new Map<string, Set<string>>();
  private readonly stopwords = new Set([
    'the',
    'of',
    'and',
    'or',
    'in',
    'new',
    'years',
    'future',
    'extra',
    'booster',
    'collection',
    'set',
    'edition',
    'memorial',
    'anime',
    'card',
    'cards',
    'premium',
    'limited',
    'starter',
  ]);

  constructor() {
    this.load();
  }

  listLeaderMetadata(): LeaderKnowledgeMetadata[] {
    return Array.from(this.leaders.values()).map((entry) => ({
      id: entry.id,
      name: entry.name,
      colors: [...entry.colors],
      subtypes: [...entry.subtypes],
      setName: entry.setName,
      keywords: Array.from(entry.keywords),
    }));
  }

  getLeaderMetadata(leaderId: string): LeaderKnowledgeMetadata | null {
    const entry = this.leaders.get(leaderId.toUpperCase());
    if (!entry) {
      return null;
    }

    return {
      id: entry.id,
      name: entry.name,
      colors: [...entry.colors],
      subtypes: [...entry.subtypes],
      setName: entry.setName,
      keywords: Array.from(entry.keywords),
    };
  }

  getLeaderIdsByColor(color: string): string[] {
    const normalized = color.trim().toLowerCase();
    const ids = this.colorIndex.get(normalized);
    return ids ? Array.from(ids) : [];
  }

  matchPrompt(prompt: string): { keywords: Set<string>; leaderIds: Set<string> } {
    const normalized = this.normalize(prompt);
    const keywordMatches = new Set<string>();
    const leaderIds = new Set<string>();

    for (const [keyword, ids] of this.keywordToLeaderIds.entries()) {
      if (!keyword.length) {
        continue;
      }
      if (normalized.includes(keyword)) {
        keywordMatches.add(keyword);
        ids.forEach((id) => leaderIds.add(id));
      }
    }

    return { keywords: keywordMatches, leaderIds };
  }

  getLeaderIdsForName(name: string): string[] {
    const normalized = this.normalize(name);
    const ids = this.nameToLeaderIds.get(normalized);
    return ids ? Array.from(ids) : [];
  }

  getLeaderIdsForKeyword(keyword: string): string[] {
    const normalized = this.normalize(keyword);
    const ids = this.keywordToLeaderIds.get(normalized);
    return ids ? Array.from(ids) : [];
  }

  keywordMatchesLeader(keyword: string, leaderId: string): boolean {
    const normalized = this.normalize(keyword);
    const ids = this.keywordToLeaderIds.get(normalized);
    return ids ? ids.has(leaderId.toUpperCase()) : false;
  }

  private load(): void {
    try {
      // Use same pattern as CardDataRepository: go up 3 levels from dist/src/ai to project root
      const filePath = join(__dirname, '../../../data/leaders.json');
      this.logger.debug(`Loading leaders from: ${filePath}`);
      const raw = readFileSync(filePath, 'utf8');
      const leaders = JSON.parse(raw) as Record<string, LeaderIndexEntry>;
      
      const totalInFile = Object.keys(leaders).length;
      this.logger.debug(`Found ${totalInFile} leader entries in JSON file`);
      
      // Check for OP13 leaders in the file
      const op13Keys = Object.keys(leaders).filter((key) => key.toUpperCase().startsWith('OP13'));
      this.logger.debug(`OP13 leader keys in file: ${op13Keys.length} (${op13Keys.slice(0, 3).join(', ')})`);

      Object.values(leaders).forEach((leader) => {
        const leaderId = leader.id.toUpperCase();
        const keywordSet = new Set<string>();

        this.addNameAlias(leader.name, leaderId, keywordSet);
        this.tokenize(leader.name).forEach((token) => this.addNameAlias(token, leaderId, keywordSet));

        leader.subtypes.forEach((subtype) => {
          this.addKeyword(subtype, leaderId, keywordSet);
          this.tokenize(subtype).forEach((token) => this.addKeyword(token, leaderId, keywordSet));
        });

        leader.colors.forEach((color) => {
          this.addKeyword(color, leaderId, keywordSet);
          this.indexColor(color, leaderId);
        });

        if (leader.setName) {
          this.addKeyword(leader.setName, leaderId, keywordSet);
          this.tokenize(leader.setName).forEach((token) => this.addKeyword(token, leaderId, keywordSet));
        }

        this.leaders.set(leaderId, {
          id: leaderId,
          name: leader.name,
          colors: leader.colors.map((color) => color.trim().toLowerCase()),
          subtypes: leader.subtypes.map((subtype) => subtype.trim()),
          setName: leader.setName ?? null,
          keywords: keywordSet,
        });
      });

      this.logger.log(
        `Loaded leader knowledge with ${this.leaders.size} leaders, ${this.keywordToLeaderIds.size} keywords and ${this.nameToLeaderIds.size} names`,
      );
      
      // Verify OP13 leaders were loaded
      const loadedOp13 = Array.from(this.leaders.keys()).filter((id) => id.startsWith('OP13'));
      if (op13Keys.length > 0 && loadedOp13.length === 0) {
        this.logger.warn(`OP13 leaders found in file (${op13Keys.length}) but not loaded!`);
      } else if (loadedOp13.length > 0) {
        this.logger.debug(`OP13 leaders loaded: ${loadedOp13.length} (${loadedOp13.slice(0, 3).join(', ')})`);
      }
    } catch (error) {
      this.logger.error(`Failed to load leader knowledge: ${(error as Error).message}`);
    }
  }

  private addNameAlias(value: string, leaderId: string, keywordsSet?: Set<string>): void {
    const normalized = this.normalize(value);
    if (!normalized || normalized.length < 3 || this.stopwords.has(normalized)) {
      return;
    }
    const upperId = leaderId.toUpperCase();
    const bucket = this.nameToLeaderIds.get(normalized) ?? new Set<string>();
    bucket.add(upperId);
    this.nameToLeaderIds.set(normalized, bucket);
    keywordsSet?.add(normalized);
    this.addKeyword(normalized, upperId, keywordsSet); // also searchable as keyword
  }

  private addKeyword(value: string, leaderId: string, keywordsSet?: Set<string>): void {
    const normalized = this.normalize(value);
    if (!normalized || normalized.length < 3 || this.stopwords.has(normalized)) {
      return;
    }
    const upperId = leaderId.toUpperCase();
    const bucket = this.keywordToLeaderIds.get(normalized) ?? new Set<string>();
    bucket.add(upperId);
    this.keywordToLeaderIds.set(normalized, bucket);
    keywordsSet?.add(normalized);
  }

  private tokenize(value: string): string[] {
    return this.normalize(value)
      .split(' ')
      .filter((part) => part.length >= 3 && !this.stopwords.has(part));
  }

  private normalize(value: string): string {
    return value.toLowerCase().replace(/[^a-z0-9\s]/g, ' ').replace(/\s+/g, ' ').trim();
  }

  private indexColor(color: string, leaderId: string): void {
    const normalized = color.trim().toLowerCase();
    if (!normalized) {
      return;
    }
    const bucket = this.colorIndex.get(normalized) ?? new Set<string>();
    bucket.add(leaderId.toUpperCase());
    this.colorIndex.set(normalized, bucket);
  }
}
