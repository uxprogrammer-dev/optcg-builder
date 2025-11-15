import { Injectable, InternalServerErrorException, Logger } from '@nestjs/common';
import { DeckCardSuggestionWithCard, GeneratedDeck } from '@shared/contracts/ai';
import { OptcgCard } from '@shared/types/optcg-card';
import archiver, { type Archiver } from 'archiver';
import axios from 'axios';
import { PassThrough, Readable } from 'stream';

@Injectable()
export class FilesService {
  private readonly logger = new Logger(FilesService.name);

  async createDeckArchive(deck: GeneratedDeck & { totalCards: number }): Promise<{
    stream: Readable;
    fileName: string;
  }> {
    const archiveStream = new PassThrough();
    const archive = archiver('zip', {
      zlib: { level: 9 },
    });

    archive.on('error', (error: Error) => {
      this.logger.error('Failed to build archive', error as Error);
      archiveStream.destroy(error as Error);
    });

    archive.pipe(archiveStream);

    const tasks: Promise<void>[] = [];

    tasks.push(
      this.appendCardImage(archive, deck.leader, 'leader'),
    );

    for (const deckCard of deck.cards) {
      tasks.push(this.appendDeckCard(archive, deckCard));
    }

    // Start building the archive asynchronously and return the stream immediately.
    // This prevents buffering the entire ZIP in memory before sending.
    (async () => {
      try {
        await Promise.all(tasks);
        await archive.finalize();
      } catch (error) {
        this.logger.error('Unable to finalize deck archive', error as Error);
        archiveStream.destroy(error as Error);
      }
    })().catch((error) => {
      this.logger.error('Archive pipeline error', error as Error);
      archiveStream.destroy(error as Error);
    });

    const fileName = `optcg-deck-${deck.deckId}.zip`;
    return { stream: archiveStream, fileName };
  }

  private async appendDeckCard(
    archive: Archiver,
    deckCard: DeckCardSuggestionWithCard,
  ): Promise<void> {
    const fileName = this.sanitizeFileName(
      `cards/${deckCard.card.id}-${deckCard.card.name}-x${deckCard.quantity}.jpg`,
    );
    await this.appendImageToArchive(archive, deckCard.card.imageUrl, fileName);
  }

  private async appendCardImage(
    archive: Archiver,
    card: OptcgCard,
    folder: string,
  ): Promise<void> {
    const fileName = this.sanitizeFileName(`${folder}/${card.id}-${card.name}.jpg`);
    await this.appendImageToArchive(archive, card.imageUrl, fileName);
  }

  private async appendImageToArchive(
    archive: Archiver,
    imageUrl: string,
    fileName: string,
  ): Promise<void> {
    try {
      const response = await axios.get<ArrayBuffer>(imageUrl, { responseType: 'arraybuffer' });
      archive.append(Buffer.from(response.data), { name: fileName });
    } catch (error) {
      this.logger.warn(`Unable to fetch image ${imageUrl}: ${(error as Error).message}`);
    }
  }

  private sanitizeFileName(fileName: string): string {
    return fileName.replace(/[^a-zA-Z0-9\-_/\. ]/g, '').replace(/\s+/g, '_');
  }
}

