import { Module } from '@nestjs/common';
import { DecksController } from './decks.controller';
import { DecksService } from './decks.service';
import { DeckRegistryService } from './deck-registry.service';
import { AiModule } from '../ai/ai.module';
import { OptcgModule } from '../optcg/optcg.module';
import { FilesModule } from '../files/files.module';

@Module({
  imports: [AiModule, OptcgModule, FilesModule],
  controllers: [DecksController],
  providers: [DecksService, DeckRegistryService],
  exports: [DecksService, DeckRegistryService],
})
export class DecksModule {}

