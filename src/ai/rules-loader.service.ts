import { Injectable, Logger } from '@nestjs/common';
import { promises as fs } from 'fs';
import { join } from 'path';

type RulesSection = {
  title: string;
  paragraphs: string[];
};

type RulesSource = {
  path: string;
  page_count: number;
  sections: RulesSection[];
  text?: string;
};

type RulesPayload = {
  version: string;
  extracted_date?: string;
  summary?: string;
  sources: Record<string, RulesSource>;
};

type PromptFormattingOptions = {
  maxSources?: number;
  maxSectionsPerSource?: number;
  maxParagraphsPerSection?: number;
  includeSummary?: boolean;
};

const FALLBACK_RULES = [
  'One Piece TCG Core Rules:',
  '- Each player starts with 1 Leader (5000 power) and 5 Life cards.',
  '- Decks contain exactly 50 cards (Leader excluded) with max 4 copies per card.',
  '- Gain 1 DON!! card each turn (up to 10); DON!! can be attached for +1000 power or spent for costs.',
  '- Characters and Leaders alternate between Active (upright) and Rested (tapped) states.',
  '- Battle sequence: Attack declaration → Block step → Counter step → Damage → End.',
  '- Use [Blocker] characters in the Block step; Counter cards are played from hand in the Counter step.',
  '- Life cards are added to hand when damage is taken; victory when opponent\'s Life reaches 0.',
  '- Respect leader color identity when building decks; only those colors (or multicolor pairs) are legal.',
  '- Common keywords: [Rush], [Double Attack], [Trigger], [On Play], [Activate: Main], [Once Per Turn].',
].join('\n');

@Injectable()
export class RulesLoaderService {
  private readonly logger = new Logger(RulesLoaderService.name);
  private readonly rulesPath: string;
  private cache: RulesPayload | null = null;
  private loadingPromise: Promise<RulesPayload | null> | null = null;

  constructor() {
    const manualDir = join(process.cwd(), 'data', 'manual');
    this.rulesPath = join(manualDir, 'rules_extracted.json');
  }

  /**
   * Load rules JSON from disk (cached after the first read).
   */
  async loadRules(force = false): Promise<RulesPayload | null> {
    if (!force && this.cache) {
      return this.cache;
    }

    if (this.loadingPromise) {
      return this.loadingPromise;
    }

    this.loadingPromise = this.readRulesFromDisk()
      .then((payload) => {
        this.cache = payload;
        return payload;
      })
      .finally(() => {
        this.loadingPromise = null;
      });

    return this.loadingPromise;
  }

  /**
   * Returns a formatted block of rules suitable for inclusion in OpenAI prompts.
   * Falls back to a small hard-coded primer if the extracted rules are unavailable.
   */
  async getRulesForPrompt(options: PromptFormattingOptions = {}): Promise<string> {
    const payload = await this.loadRules();
    if (!payload) {
      return FALLBACK_RULES;
    }

    const {
      maxSources = 2,
      maxSectionsPerSource = 4,
      maxParagraphsPerSection = 2,
      includeSummary = true,
    } = options;

    const selectedSources = Object.entries(payload.sources).slice(0, maxSources);
    const lines: string[] = [];

    if (includeSummary && payload.summary) {
      lines.push('Rules Summary:', payload.summary.trim(), '');
    }

    for (const [sourceName, source] of selectedSources) {
      lines.push(`${this.formatSourceName(sourceName)} (${source.page_count} pages):`);

      const sections = (source.sections ?? []).slice(0, maxSectionsPerSource);
      for (const section of sections) {
        if (!section || !section.title) continue;
        lines.push(`- ${section.title}`);

        const paragraphs = (section.paragraphs ?? []).slice(0, maxParagraphsPerSection);
        for (const paragraph of paragraphs) {
          const trimmed = paragraph.replace(/\s+/g, ' ').trim();
          if (!trimmed) continue;
          lines.push(`  • ${trimmed}`);
        }
      }

      lines.push('');
    }

    if (!lines.length) {
      return FALLBACK_RULES;
    }

    return lines.join('\n').trim();
  }

  /**
   * Returns a concise summary for UI display or fallback to the hard-coded rules primer.
   */
  async getRulesSummary(): Promise<string> {
    const payload = await this.loadRules();
    if (!payload) {
      return FALLBACK_RULES;
    }

    if (payload.summary && payload.summary.trim().length > 0) {
      return payload.summary.trim();
    }

    // Synthesise a quick summary from the first few sections if explicit summary is missing.
    const sections = Object.values(payload.sources ?? {})
      .flatMap((source) => source.sections ?? [])
      .slice(0, 3);

    const lines: string[] = [];
    for (const section of sections) {
      if (!section.title) continue;
      lines.push(section.title);
      const firstParagraph = (section.paragraphs ?? []).find((paragraph) => paragraph && paragraph.trim());
      if (firstParagraph) {
        lines.push(firstParagraph.replace(/\s+/g, ' ').trim());
      }
    }

    return lines.length > 0 ? lines.join(' • ') : FALLBACK_RULES;
  }

  private async readRulesFromDisk(): Promise<RulesPayload | null> {
    try {
      const raw = await fs.readFile(this.rulesPath, 'utf-8');
      const payload = JSON.parse(raw) as RulesPayload;

      if (!payload || typeof payload !== 'object' || !payload.sources) {
        this.logger.warn(`Rules file at ${this.rulesPath} is missing expected fields.`);
        return null;
      }

      return payload;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
        this.logger.warn(`Rules file not found at ${this.rulesPath}. Falling back to primer.`);
        return null;
      }

      this.logger.error(`Failed to read rules file at ${this.rulesPath}: ${(error as Error).message}`, error as Error);
      return null;
    }
  }

  private formatSourceName(name: string): string {
    return name
      .split(/[_\-]/g)
      .map((chunk) => chunk.charAt(0).toUpperCase() + chunk.slice(1))
      .join(' ');
  }

  getFallbackRules(): string {
    return FALLBACK_RULES;
  }
}

