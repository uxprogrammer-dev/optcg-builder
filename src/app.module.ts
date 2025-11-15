import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import configuration from './config/configuration';
import { validationSchema } from './config/validation';
import { AiModule } from './ai/ai.module';
import { DecksModule } from './decks/decks.module';
import { FilesModule } from './files/files.module';
import { OptcgModule } from './optcg/optcg.module';
import { ServeStaticModule } from '@nestjs/serve-static';
import { existsSync } from 'fs';
import { join } from 'path';

const staticClientModules = (() => {
  const clientDist = join(__dirname, '..', 'apps', 'web', 'out');
  if (!existsSync(clientDist)) {
    return [];
  }

  return [
    ServeStaticModule.forRoot({
      rootPath: clientDist,
      exclude: ['/api*'],
    }),
  ];
})();

@Module({
  imports: [
    ConfigModule.forRoot({
      isGlobal: true,
      load: [configuration],
      validationSchema,
    }),
    ...staticClientModules,
    OptcgModule,
    AiModule,
    DecksModule,
    FilesModule,
  ],
})
export class AppModule {}

