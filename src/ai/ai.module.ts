import { Module } from '@nestjs/common';
import { AiController } from './ai.controller';
import { AiService } from './ai.service';
import { OpenAiService } from './openai.service';
import { OptcgModule } from '../optcg/optcg.module';
import { LeaderKnowledgeService } from './leader-knowledge.service';
import { CardKnowledgeService } from './card-knowledge.service';
import { MlIntentService } from './ml-intent.service';
import { LocalDeckBuilderService } from './local-deck-builder.service';
import { MlDeckService } from './ml-deck.service';
import { TournamentSynergyService } from './tournament-synergy.service';
import { RulesLoaderService } from './rules-loader.service';
import { ArchetypeService } from './archetype.service';
import { CardTierService } from './card-tier.service';
import { LeaderStrategyService } from './leader-strategy.service';
import { ProgressService } from './progress.service';

@Module({
  imports: [OptcgModule],
  controllers: [AiController],
      providers: [
        AiService,
        OpenAiService,
        LeaderKnowledgeService,
        CardKnowledgeService,
        MlIntentService,
        LocalDeckBuilderService,
        MlDeckService,
        TournamentSynergyService,
        RulesLoaderService,
        ArchetypeService,
        CardTierService,
        LeaderStrategyService,
        ProgressService,
      ],
      exports: [
        AiService,
        CardKnowledgeService,
        OpenAiService,
        RulesLoaderService,
        TournamentSynergyService,
        ArchetypeService,
        CardTierService,
        LeaderStrategyService,
        ProgressService,
      ],
})
export class AiModule {}
