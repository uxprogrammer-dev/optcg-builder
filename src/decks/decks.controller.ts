import { Body, Controller, Get, Param, Post, Res, StreamableFile } from '@nestjs/common';
import { Response } from 'express';
import { DecksService } from './decks.service';
import { GenerateDeckDto } from './dto/generate-deck.dto';
import { FilesService } from '../files/files.service';
import { GeneratedDeck } from '@shared/contracts/ai';
import { SuggestCardDto } from './dto/suggest-card.dto';
import { ReviewDeckDto } from './dto/review-deck.dto';
import { ProgressService } from '../ai/progress.service';

@Controller('decks')
export class DecksController {
  constructor(
    private readonly decksService: DecksService,
    private readonly filesService: FilesService,
    private readonly progressService: ProgressService,
  ) {}

  @Post()
  async generateDeck(@Body() generateDeckDto: GenerateDeckDto) {
    const progressId = generateDeckDto.progressId || `deck-${Date.now()}-${Math.random().toString(36).substring(7)}`;
    
    console.log('[DecksController] Received request:', {
      prompt: generateDeckDto.prompt?.substring(0, 50),
      leaderCardId: generateDeckDto.leaderCardId,
      useOpenAi: generateDeckDto.useOpenAi,
      setIds: generateDeckDto.setIds,
      metaOnly: generateDeckDto.metaOnly,
      tournamentOnly: generateDeckDto.tournamentOnly,
      progressId,
    });
    
    // Initialize progress tracking
    this.progressService.createProgress(progressId, 'deck');
    
    try {
      const deck = await this.decksService.generateDeck({
        ...generateDeckDto,
        progressId,
        progressService: this.progressService,
      });
      this.progressService.completeProgress(progressId, `Generated deck with ${deck.cards.length} unique cards`);
      return { ...deck, progressId };
    } catch (error) {
      this.progressService.failProgress(progressId, (error as Error).message);
      throw error;
    }
  }

  @Get(':deckId')
  async getDeck(@Param('deckId') deckId: string) {
    return this.decksService.getDeck(deckId);
  }

  @Get(':deckId/download')
  async downloadDeck(
    @Param('deckId') deckId: string,
    @Res({ passthrough: true }) res: Response,
  ): Promise<StreamableFile> {
    const deck = this.decksService.getDeck(deckId);
    const { stream, fileName } = await this.filesService.createDeckArchive(deck);

    res.set({
      'Content-Type': 'application/zip',
      'Content-Disposition': `attachment; filename="${fileName}"`,
    });

    return new StreamableFile(stream);
  }

  @Post('download')
  async downloadDeckFromPayload(
    @Body() deck: GeneratedDeck,
    @Res({ passthrough: true }) res: Response,
  ): Promise<StreamableFile> {
    const totalCards = deck.cards.reduce((sum, card) => sum + card.quantity, 0);
    const { stream, fileName } = await this.filesService.createDeckArchive({
      ...deck,
      totalCards,
    });

    res.set({
      'Content-Type': 'application/zip',
      'Content-Disposition': `attachment; filename="${fileName}"`,
    });

    return new StreamableFile(stream);
  }

  @Post('cards/suggest')
  async suggestCards(@Body() dto: SuggestCardDto) {
    return this.decksService.suggestCards(dto);
  }

  @Post('review')
  async reviewDeck(@Body() dto: ReviewDeckDto) {
    return this.decksService.reviewDeck(dto);
  }
}

