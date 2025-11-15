import { Injectable, Logger } from '@nestjs/common';
import { LeaderKnowledgeService, LeaderKnowledgeMetadata } from './leader-knowledge.service';
import { CardKnowledgeService } from './card-knowledge.service';

export type LeaderIntentMatch = {
  leaderId: string;
  leaderName: string;
  score: number;
  matchedKeywords: string[];
  matchedColors: string[];
  reasons: string[];
  metadata: LeaderKnowledgeMetadata;
};

export type ColorHint = {
  color: string;
  score: number;
};

export type PromptIntentAnalysis = {
  prompt: string;
  normalizedPrompt: string;
  tokens: string[];
  keywords: string[];
  cardIds: string[];
  colors: string[];
  colorHints: ColorHint[];
  explicitLeaderIds: string[];
  leaderMatches: LeaderIntentMatch[];
  unmatchedKeywords: string[];
};

@Injectable()
export class IntentMatcherService {
  private readonly logger = new Logger(IntentMatcherService.name);
  private readonly colorWordMap = new Map<string, string>([
    ['red', 'red'],
    ['blue', 'blue'],
    ['green', 'green'],
    ['yellow', 'yellow'],
    ['purple', 'purple'],
    ['black', 'black'],
  ]);
  private readonly stopwords = new Set<string>([
    'the',
    'of',
    'and',
    'or',
    'to',
    'in',
    'a',
    'an',
    'deck',
    'around',
    'strategy',
    'strategies',
    'build',
    'describe',
    'competitive',
    'characters',
    'character',
    'leader',
    'leaders',
    'cards',
    'card',
    'show',
    'download',
    'images',
    'please',
    'make',
    'want',
    'focus',
    'based',
    'focusing',
    'single',
    'archive',
    'list',
    'let',
    'us',
    'using',
  ]);

  constructor(
    private readonly leaderKnowledge: LeaderKnowledgeService,
    private readonly cardKnowledge: CardKnowledgeService,
  ) {}

  async analyzePrompt(prompt: string): Promise<PromptIntentAnalysis> {
    const normalizedPrompt = this.normalize(prompt);
    const tokens = this.tokenize(normalizedPrompt);
    const tokenSet = new Set(tokens);

    const directColors = new Set<string>();
    tokens.forEach((token) => {
      const color = this.colorWordMap.get(token);
      if (color) {
        directColors.add(color);
      }
    });

    const cardIdMatches = (prompt.match(/(?:OP|ST|EB)\d{2}-\d{3}/gi) ?? []).map((id) => id.toUpperCase());

    const knowledgeMatches = this.leaderKnowledge.matchPrompt(prompt);
    const knowledgeKeywordSet = new Set<string>(knowledgeMatches.keywords);

    const leaderMetadataList = this.leaderKnowledge.listLeaderMetadata();

    const explicitLeaderIds = new Set<string>();

    // Direct name mentions (full name or all tokens present)
    for (const metadata of leaderMetadataList) {
      const normalizedName = this.normalize(metadata.name);
      if (!normalizedName) {
        continue;
      }
      const nameTokens = this.tokenize(metadata.name);
      const hasAllTokens = nameTokens.length > 0 && nameTokens.every((token) => tokenSet.has(token));
      if (normalizedPrompt.includes(normalizedName) || hasAllTokens) {
        explicitLeaderIds.add(metadata.id.toUpperCase());
      }
    }

    cardIdMatches.forEach((id) => {
      if (this.leaderKnowledge.getLeaderMetadata(id)) {
        explicitLeaderIds.add(id);
      }
    });

    const derivedColorScores = await this.deriveColorsFromKeywords(tokenSet);
    derivedColorScores.forEach((score, color) => {
      if (score > 0) {
        directColors.add(color);
      }
    });

    const leaderMatches = this.rankLeaders({
      leaderMetadataList,
      normalizedPrompt,
      promptTokens: tokenSet,
      matchedKeywords: knowledgeKeywordSet,
      matchedColors: directColors,
      explicitLeaderIds,
    });

    const consumedKeywords = new Set<string>();
    leaderMatches.forEach((match) => match.matchedKeywords.forEach((keyword) => consumedKeywords.add(keyword)));

    const unmatchedKeywords = Array.from(knowledgeKeywordSet).filter((keyword) => !consumedKeywords.has(keyword));

    const colorHints: ColorHint[] = Array.from(derivedColorScores.entries())
      .map(([color, score]) => ({ color, score }))
      .sort((a, b) => b.score - a.score);

    return {
      prompt,
      normalizedPrompt,
      tokens: Array.from(tokenSet),
      keywords: Array.from(knowledgeKeywordSet),
      cardIds: cardIdMatches,
      colors: Array.from(directColors),
      colorHints,
      explicitLeaderIds: Array.from(explicitLeaderIds),
      leaderMatches,
      unmatchedKeywords,
    };
  }

  private rankLeaders({
    leaderMetadataList,
    normalizedPrompt,
    promptTokens,
    matchedKeywords,
    matchedColors,
    explicitLeaderIds,
  }: {
    leaderMetadataList: LeaderKnowledgeMetadata[];
    normalizedPrompt: string;
    promptTokens: Set<string>;
    matchedKeywords: Set<string>;
    matchedColors: Set<string>;
    explicitLeaderIds: Set<string>;
  }): LeaderIntentMatch[] {
    const matches: LeaderIntentMatch[] = [];

    for (const metadata of leaderMetadataList) {
      const keywordSet = new Set<string>();

      metadata.keywords.forEach((keyword) => {
        if (
          matchedKeywords.has(keyword) ||
          this.keywordAppearsInPrompt(keyword, normalizedPrompt, promptTokens)
        ) {
          keywordSet.add(keyword);
        }
      });

      const colorHits = metadata.colors.filter((color) => matchedColors.has(color));

      let score = keywordSet.size * 2 + colorHits.length * 4;
      const reasons: string[] = [];

      if (keywordSet.size) {
        reasons.push(`keywords: ${Array.from(keywordSet).join(', ')}`);
      }

      if (colorHits.length) {
        reasons.push(`colors: ${colorHits.join(', ')}`);
      }

      if (normalizedPrompt.includes(this.normalize(metadata.name))) {
        score += 10;
        reasons.push('leader name mentioned');
      }

      if (explicitLeaderIds.has(metadata.id)) {
        score += 25;
        reasons.push('explicit leader reference');
      }

      if (score > 0) {
        matches.push({
          leaderId: metadata.id,
          leaderName: metadata.name,
          score,
          matchedKeywords: Array.from(keywordSet),
          matchedColors: colorHits,
          reasons,
          metadata,
        });
      }
    }

    matches.sort((a, b) => b.score - a.score);
    return matches;
  }

  private async deriveColorsFromKeywords(tokens: Set<string>): Promise<Map<string, number>> {
    const scores = new Map<string, number>();

    for (const token of tokens) {
      if (this.stopwords.has(token) || this.colorWordMap.has(token) || token.length < 3) {
        continue;
      }

      try {
        const cards = await this.cardKnowledge.findByKeyword(token);
        cards
          .slice(0, 20)
          .forEach((card) => {
            card.colors.forEach((color) => {
              const normalizedColor = color.toLowerCase();
              const previous = scores.get(normalizedColor) ?? 0;
              scores.set(normalizedColor, previous + 1);
            });
          });
      } catch (error) {
        this.logger.warn(`Failed to derive colors for token "${token}": ${(error as Error).message}`);
      }
    }

    return scores;
  }

  private keywordAppearsInPrompt(keyword: string, normalizedPrompt: string, tokens: Set<string>): boolean {
    if (!keyword) {
      return false;
    }
    if (keyword.includes(' ')) {
      return normalizedPrompt.includes(keyword);
    }
    return tokens.has(keyword);
  }

  private normalize(value: string): string {
    return value.toLowerCase().replace(/[^a-z0-9\s]/g, ' ').replace(/\s+/g, ' ').trim();
  }

  private tokenize(value: string): string[] {
    return value
      .split(' ')
      .map((part) => part.trim())
      .filter((part) => part.length > 0 && !this.stopwords.has(part));
  }
}
