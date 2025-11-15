import { DeckCharacteristicId } from '@shared/contracts/ai';
import { IsArray, IsBoolean, IsInt, IsOptional, IsString, Max, MaxLength, Min, MinLength } from 'class-validator';
import { Type } from 'class-transformer';

export class SuggestCardDto {
  @IsString()
  @MinLength(1)
  leaderCardId!: string;

  @IsOptional()
  @IsString()
  @MinLength(1)
  @MaxLength(50)
  cardIdQuery?: string;

  @IsOptional()
  @IsString()
  characteristicId?: DeckCharacteristicId;

  @IsOptional()
  @IsArray()
  @IsString({ each: true })
  excludeCardIds?: string[];

  @IsOptional()
  @Type(() => Number)
  @IsInt()
  @Min(1)
  @Max(30)
  limit?: number;

  @IsOptional()
  @IsArray()
  @IsString({ each: true })
  setIds?: string[];

  @IsOptional()
  @IsBoolean()
  metaOnly?: boolean;
}
