import { IsBoolean, IsOptional, IsString, MinLength, IsArray } from 'class-validator';

export class PromptDto {
  @IsString()
  @MinLength(5)
  prompt!: string;

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
  @IsBoolean()
  regenerate?: boolean;

  @IsOptional()
  @IsString()
  progressId?: string;
}

