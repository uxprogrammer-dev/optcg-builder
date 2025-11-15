import { IsArray, IsBoolean, IsOptional, IsString, MinLength } from 'class-validator';

export class GenerateDeckDto {
  @IsString()
  @MinLength(3)
  prompt!: string;

  @IsString()
  @MinLength(3)
  leaderCardId!: string;

  @IsOptional()
  @IsBoolean()
  useOpenAi?: boolean;

  @IsOptional()
  @IsArray()
  @IsString({ each: true })
  setIds?: string[];

  @IsOptional()
  @IsBoolean()
  metaOnly?: boolean;

  @IsOptional()
  @IsBoolean()
  tournamentOnly?: boolean;

  @IsOptional()
  @IsString()
  progressId?: string;
}

