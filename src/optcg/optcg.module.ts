import { Module } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { HttpModule } from '@nestjs/axios';
import { OptcgApiService } from './optcg-api.service';
import { CardDataRepository } from './repositories/card-data.repository';

@Module({
  imports: [
    HttpModule.registerAsync({
      inject: [ConfigService],
      useFactory: (configService: ConfigService) => ({
        baseURL: configService.get<string>('optcgApi.baseUrl'),
        timeout: configService.get<number>('optcgApi.timeout'),
        headers: {
          Accept: 'application/json',
        },
      }),
    }),
  ],
  providers: [OptcgApiService, CardDataRepository],
  exports: [OptcgApiService, CardDataRepository],
})
export class OptcgModule {}

