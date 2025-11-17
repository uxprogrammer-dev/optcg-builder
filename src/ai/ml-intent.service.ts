import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { spawn } from 'child_process';
import { delimiter, normalize, resolve, dirname, join } from 'path';
import { existsSync, readFileSync } from 'fs';
import {
  LeaderIntentMatch,
  PromptIntentAnalysis,
  ColorHint,
} from './intent-matcher.service';
import { LeaderKnowledgeService, LeaderKnowledgeMetadata } from './leader-knowledge.service';

interface MlIntentConfig {
  enabled: boolean;
  pythonPath: string;
  module: string;
  modelPath: string;
  promptVocabularyPath: string;
  colorVocabularyPath: string;
  keywordVocabularyPath: string;
  leaderVocabularyPath: string;
  strategyVocabularyPath: string;
  leaderIdsPath: string;
  dataRoot: string;
  timeoutMs: number;
  threshold: number;
  topKLeaders: number;
  topKKeywords: number;
}

interface MlIntentPayload {
  prompt: string;
  normalizedPrompt: string;
  tokens: string[];
  keywords: string[];
  cardIds: string[];
  colors: string[];
  colorHints: ColorHint[];
  explicitLeaderIds: string[];
  leaderMatches: Array<{
    leaderId: string;
    leaderName: string;
    score: number;
    matchedKeywords: string[];
    matchedColors: string[];
    reasons: string[];
  }>;
  unmatchedKeywords: string[];
}

@Injectable()
export class MlIntentService {
  private readonly logger = new Logger(MlIntentService.name);
  private readonly config: MlIntentConfig;

  constructor(
    private readonly configService: ConfigService,
    private readonly leaderKnowledgeService: LeaderKnowledgeService,
  ) {
    this.config = {
      enabled: this.configService.get<boolean>('mlIntent.enabled') ?? false,
      pythonPath: this.configService.get<string>('mlIntent.pythonPath') ?? this.configService.get<string>('mlModel.pythonPath') ?? 'python',
      module: this.configService.get<string>('mlIntent.module') ?? 'ml.inference.intent_classify',
      modelPath: this.configService.get<string>('mlIntent.modelPath') ?? '',
      promptVocabularyPath: this.configService.get<string>('mlIntent.promptVocabularyPath') ?? '',
      colorVocabularyPath: this.configService.get<string>('mlIntent.colorVocabularyPath') ?? '',
      keywordVocabularyPath: this.configService.get<string>('mlIntent.keywordVocabularyPath') ?? '',
      leaderVocabularyPath: this.configService.get<string>('mlIntent.leaderVocabularyPath') ?? '',
      strategyVocabularyPath: this.configService.get<string>('mlIntent.strategyVocabularyPath') ?? '',
      leaderIdsPath: this.configService.get<string>('mlIntent.leaderIdsPath') ?? '',
      dataRoot: this.configService.get<string>('mlIntent.dataRoot') ?? this.configService.get<string>('mlModel.dataRoot') ?? 'data',
      timeoutMs: this.configService.get<number>('mlIntent.timeoutMs') ?? 10000,
      threshold: this.configService.get<number>('mlIntent.threshold') ?? 0.5,
      topKLeaders: this.configService.get<number>('mlIntent.topKLeaders') ?? 10,
      topKKeywords: this.configService.get<number>('mlIntent.topKKeywords') ?? 20,
    };

    if (this.isEnabled) {
      this.logger.log(
        `ML Intent Service enabled: model=${this.config.modelPath}`,
      );
    } else {
      this.logger.warn(
        `ML Intent Service disabled: enabled=${this.config.enabled}, modelPath=${this.config.modelPath ? 'set' : 'missing'}`,
      );
    }
  }

  get isEnabled(): boolean {
    return (
      this.config.enabled &&
      Boolean(this.config.modelPath) &&
      this.canDeriveVocabPaths()
    );
  }

  /**
   * Derives vocabulary paths from the model path.
   * Model path format: models/intent_run_YYYYMMDD-HHMMSS/intent_classifier.keras
   * Vocab paths: models/intent_run_YYYYMMDD-HHMMSS/vocab/*.json
   */
  private deriveVocabPaths(modelPath: string): {
    promptVocab: string;
    colorVocab: string;
    keywordVocab: string;
    leaderVocab: string;
    strategyVocab: string;
    leaderIds: string;
  } {
    const modelDir = dirname(this.normalizePath(modelPath));
    return {
      promptVocab: join(modelDir, 'vocab', 'prompt_vocabulary.txt'),
      colorVocab: join(modelDir, 'vocab', 'color_vocabulary.json'),
      keywordVocab: join(modelDir, 'vocab', 'keyword_vocabulary.json'),
      leaderVocab: join(modelDir, 'vocab', 'leader_vocabulary.json'),
      strategyVocab: join(modelDir, 'vocab', 'strategy_vocabulary.json'),
      leaderIds: join(modelDir, 'vocab', 'leader_ids.json'),
    };
  }

  /**
   * Checks if vocab paths can be derived from the model path.
   */
  private canDeriveVocabPaths(): boolean {
    if (!this.config.modelPath) {
      return false;
    }
    const derived = this.deriveVocabPaths(this.config.modelPath);
    return (
      existsSync(this.normalizePath(derived.promptVocab)) &&
      existsSync(this.normalizePath(derived.colorVocab)) &&
      existsSync(this.normalizePath(derived.keywordVocab)) &&
      existsSync(this.normalizePath(derived.leaderVocab)) &&
      existsSync(this.normalizePath(derived.strategyVocab)) &&
      existsSync(this.normalizePath(derived.leaderIds))
    );
  }

  /**
   * Normalize a path for the current platform.
   */
  private normalizePath(path: string): string {
    if (!path) {
      return path;
    }
    // Handle absolute Windows paths like C:\...
    if (path.match(/^[A-Z]:\\/i)) {
      return resolve(path);
    }
    // Handle absolute Unix paths starting with /
    if (path.startsWith('/')) {
      // If it's an absolute path starting with / but not /c/, resolve it
      return resolve(path.replace(/\//g, '\\'));
    }
    // For relative paths or already normalized paths, use path.normalize
    return normalize(path);
  }

  /**
   * Invoke Python ML model for intent classification.
   */
  private async invokePythonModel(prompt: string): Promise<MlIntentPayload | null> {
    const pythonPath = this.normalizePath(this.config.pythonPath);
    const modelPath = this.normalizePath(this.config.modelPath);

    // Derive vocab paths from model path
    const derived = this.deriveVocabPaths(modelPath);
    const promptVocabPath = this.normalizePath(derived.promptVocab);
    const colorVocabPath = this.normalizePath(derived.colorVocab);
    const keywordVocabPath = this.normalizePath(derived.keywordVocab);
    const leaderVocabPath = this.normalizePath(derived.leaderVocab);
    const strategyVocabPath = this.normalizePath(derived.strategyVocab);
    const leaderIdsPath = this.normalizePath(derived.leaderIds);

    const args = [
      '-m',
      this.config.module,
      '--prompt',
      prompt,
      '--model',
      modelPath,
      '--prompt-vocab',
      promptVocabPath,
      '--color-vocab',
      colorVocabPath,
      '--keyword-vocab',
      keywordVocabPath,
      '--leader-vocab',
      leaderVocabPath,
      '--strategy-vocab',
      strategyVocabPath,
      '--leader-ids',
      leaderIdsPath,
      '--threshold',
      String(this.config.threshold),
      '--top-k-leaders',
      String(this.config.topKLeaders),
      '--top-k-keywords',
      String(this.config.topKKeywords),
    ];

    // Verify Python executable exists
    if (!existsSync(pythonPath)) {
      throw new Error(
        `Python executable not found at: ${pythonPath}. Please verify ML_INTENT_PYTHON_PATH in your .env file.`,
      );
    }

    return new Promise<MlIntentPayload | null>((resolve, reject) => {
      const pythonProcess = spawn(pythonPath, args, {
        cwd: process.cwd(),
        env: {
          ...process.env,
          PYTHONPATH: process.cwd(),
          PYTHONUNBUFFERED: '1',
        },
        stdio: ['pipe', 'pipe', 'pipe'],
      });

      let stdout = '';
      let stderr = '';

      pythonProcess.stdout.on('data', (data: Buffer) => {
        stdout += data.toString();
      });

      pythonProcess.stderr.on('data', (data: Buffer) => {
        stderr += data.toString();
      });

      const timeout = setTimeout(() => {
        pythonProcess.kill();
        reject(new Error(`ML intent classification timed out after ${this.config.timeoutMs}ms`));
      }, this.config.timeoutMs);

      pythonProcess.on('close', (code: number | null) => {
        clearTimeout(timeout);

        if (code !== 0) {
          this.logger.error(`ML intent process exited with ${code}: ${stderr}`);
          reject(new Error(`ML intent classification failed: ${stderr}`));
          return;
        }

        try {
          // Parse JSON output - extract JSON object from stdout (handle trailing characters)
          const trimmed = stdout.trim();
          // Find the JSON object by looking for the first { and last }
          const firstBrace = trimmed.indexOf('{');
          const lastBrace = trimmed.lastIndexOf('}');
          
          if (firstBrace === -1 || lastBrace === -1 || lastBrace <= firstBrace) {
            throw new Error('No valid JSON object found in output');
          }
          
          const jsonStr = trimmed.substring(firstBrace, lastBrace + 1);
          const payload = JSON.parse(jsonStr) as MlIntentPayload;
          resolve(payload);
        } catch (error) {
          this.logger.error(`Failed to parse ML intent output: ${error}\nOutput: ${stdout}`);
          reject(new Error(`Failed to parse ML intent output: ${(error as Error).message}`));
        }
      });

      pythonProcess.on('error', (error: Error) => {
        clearTimeout(timeout);
        this.logger.error(`ML intent process error: ${error.message}`);
        reject(error);
      });
    });
  }

  /**
   * Analyze prompt and return intent analysis.
   */
  async analyzePrompt(prompt: string): Promise<PromptIntentAnalysis> {
    if (!this.isEnabled) {
      this.logger.debug('ML intent analysis skipped: service not enabled or misconfigured');
      // Return empty analysis as fallback
      return this.createEmptyAnalysis(prompt);
    }

    try {
      const mlOutput = await this.invokePythonModel(prompt);
      if (!mlOutput) {
        return this.createEmptyAnalysis(prompt);
      }

      // Convert ML output to PromptIntentAnalysis format
      return this.convertMlOutputToAnalysis(mlOutput);
    } catch (error) {
      this.logger.error(`ML intent analysis failed: ${(error as Error).message}`);
      // Return empty analysis as fallback
      return this.createEmptyAnalysis(prompt);
    }
  }

  /**
   * Convert ML model output to PromptIntentAnalysis format.
   */
  private convertMlOutputToAnalysis(mlOutput: MlIntentPayload): PromptIntentAnalysis {
    // Enrich leader matches with metadata
    const leaderMatches: LeaderIntentMatch[] = mlOutput.leaderMatches.map((match) => {
      const metadata = this.leaderKnowledgeService.getLeaderMetadata(match.leaderId);
      return {
        leaderId: match.leaderId,
        leaderName: metadata?.name || match.leaderName,
        score: match.score,
        matchedKeywords: match.matchedKeywords,
        matchedColors: match.matchedColors,
        reasons: match.reasons,
        metadata: metadata || {
          id: match.leaderId,
          name: match.leaderName,
          colors: [],
          subtypes: [],
          setName: null,
          keywords: [],
        },
      };
    });

    return {
      prompt: mlOutput.prompt,
      normalizedPrompt: mlOutput.normalizedPrompt,
      tokens: mlOutput.tokens,
      keywords: mlOutput.keywords,
      cardIds: mlOutput.cardIds,
      colors: mlOutput.colors,
      colorHints: mlOutput.colorHints,
      explicitLeaderIds: mlOutput.explicitLeaderIds,
      leaderMatches,
      unmatchedKeywords: mlOutput.unmatchedKeywords,
    };
  }

  /**
   * Create empty analysis as fallback.
   */
  private createEmptyAnalysis(prompt: string): PromptIntentAnalysis {
    const normalizedPrompt = prompt.toLowerCase().trim();
    const tokens = normalizedPrompt.split(/\s+/).filter((t) => t.length > 0);

    return {
      prompt,
      normalizedPrompt,
      tokens,
      keywords: [],
      cardIds: [],
      colors: [],
      colorHints: [],
      explicitLeaderIds: [],
      leaderMatches: [],
      unmatchedKeywords: [],
    };
  }
}

