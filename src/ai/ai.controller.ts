import { Body, Controller, Get, Param, Post } from '@nestjs/common';
import { AiService } from './ai.service';
import { PromptDto } from './dto/prompt.dto';
import { CardKnowledgeService } from './card-knowledge.service';
import { ProgressService } from './progress.service';

@Controller('ai')
export class AiController {
  constructor(
    private readonly aiService: AiService,
    private readonly cardKnowledgeService: CardKnowledgeService,
    private readonly progressService: ProgressService,
  ) {}

  @Post('leaders')
  async suggestLeaders(@Body() dto: PromptDto) {
    const { prompt, useOpenAi, setIds, metaOnly, tournamentOnly, regenerate, progressId } = dto;
    const useOpenAiValue = useOpenAi ?? false;
    const progressIdToUse = progressId || `leaders-${Date.now()}-${Math.random().toString(36).substring(7)}`;
    
    console.log('[AiController] Received request:', {
      prompt: prompt?.substring(0, 50),
      useOpenAi: useOpenAiValue,
      setIds,
      metaOnly,
      tournamentOnly,
      regenerate,
      progressId: progressIdToUse,
    });
    
    // Initialize progress tracking
    this.progressService.createProgress(progressIdToUse, 'leaders');
    
    try {
      const leaders = await this.aiService.suggestLeaders(prompt, useOpenAiValue, {
        setIds,
        metaOnly: metaOnly ?? false,
        tournamentOnly: tournamentOnly ?? false,
        regenerate: regenerate ?? false,
        progressId: progressIdToUse,
        progressService: this.progressService,
      });
      this.progressService.completeProgress(progressIdToUse, `Found ${leaders.length} leader suggestions`);
      return { prompt, leaders, progressId: progressIdToUse };
    } catch (error) {
      this.progressService.failProgress(progressIdToUse, (error as Error).message);
      throw error;
    }
  }

  @Get('sets')
  async listSets() {
    return this.cardKnowledgeService.listAvailableSets();
  }

  @Get('progress/:id')
  async getProgress(@Param('id') id: string) {
    const progress = this.progressService.getProgress(id);
    if (!progress) {
      return { id, phase: null, step: 'Not found', progress: 0 };
    }
    return progress;
  }
}

