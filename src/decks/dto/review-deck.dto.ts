import { Type } from 'class-transformer';
import { IsArray, IsInt, IsString, Max, Min, MinLength, ValidateNested } from 'class-validator';

class ReviewDeckCardDto {
  @IsString()
  @MinLength(2)
  cardSetId!: string;

  @Type(() => Number)
  @IsInt()
  @Min(1)
  @Max(4)
  quantity!: number;
}

export class ReviewDeckDto {
  @IsString()
  @MinLength(1)
  prompt!: string;

  @IsString()
  @MinLength(2)
  leaderCardId!: string;

  @IsArray()
  @ValidateNested({ each: true })
  @Type(() => ReviewDeckCardDto)
  cards!: ReviewDeckCardDto[];
}
