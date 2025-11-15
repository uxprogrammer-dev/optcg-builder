import { HttpService } from '@nestjs/axios';
import {
  Injectable,
  InternalServerErrorException,
  Logger,
  NotFoundException,
} from '@nestjs/common';
import { AxiosError } from 'axios';
import { firstValueFrom } from 'rxjs';
import { OptcgCard, OptcgCardApiResponse } from '@shared/types/optcg-card';
import { CardDataRepository } from './repositories/card-data.repository';

@Injectable()
export class OptcgApiService {
  private readonly logger = new Logger(OptcgApiService.name);

  constructor(
    private readonly httpService: HttpService,
    private readonly cardRepository: CardDataRepository,
  ) {}

  async getCardBySetId(cardSetId: string): Promise<OptcgCard> {
    try {
      const localCard = await this.cardRepository.findById(cardSetId);
      if (localCard) {
        return localCard;
      }

      const { data } = await firstValueFrom(
        this.httpService.get<OptcgCardApiResponse[]>(`/sets/card/${encodeURIComponent(cardSetId)}/`),
      );

      const [card] = data ?? [];
      if (!card) {
        throw new NotFoundException(`Card ${cardSetId} not found in OPTCG API`);
      }

      const mapped = this.mapCard(card);
      this.cardRepository.cacheCard(mapped);
      return mapped;
    } catch (error) {
      const localCard = await this.cardRepository.findById(cardSetId);
      if (localCard) {
        this.logger.warn(`Falling back to local card data for ${cardSetId}`);
        return localCard;
      }

      this.handleRequestError(error, `fetching card ${cardSetId}`);
    }
  }

  async getCardsBySetIds(cardSetIds: string[]): Promise<OptcgCard[]> {
    const uniqueIds = Array.from(new Set(cardSetIds));

    const cards = await Promise.all(uniqueIds.map((id) => this.getCardBySetId(id).catch(() => null)));
    const foundCards = cards.filter((card): card is OptcgCard => Boolean(card));

    const missingIds = uniqueIds.filter(
      (id) => !foundCards.some((card) => card.id.toLowerCase() === id.toLowerCase()),
    );

    if (missingIds.length) {
      this.logger.warn(`Missing OPTCG cards: ${missingIds.join(', ')}`);
    }

    return foundCards;
  }

  private mapCard(card: OptcgCardApiResponse): OptcgCard {
    return {
      id: card.card_set_id,
      name: card.card_name,
      text: card.card_text,
      type: card.card_type,
      color: card.card_color,
      cost: card.card_cost,
      power: card.card_power,
      life: card.life,
      attribute: card.attribute,
      subtypes: card.sub_types ? card.sub_types.split(' ') : [],
      rarity: card.rarity,
      setId: card.set_id,
      setName: card.set_name,
      imageUrl: card.card_image,
      marketPrice: card.market_price,
      inventoryPrice: card.inventory_price,
      raw: card,
    };
  }

  private handleRequestError(error: unknown, context: string): never {
    if ((error as AxiosError)?.isAxiosError) {
      const axiosError = error as AxiosError;
      this.logger.error(
        `Error ${context}: ${axiosError.message} ${(axiosError.response?.status ?? '')}`,
      );
      throw new InternalServerErrorException(
        `Unable to communicate with OPTCG API while ${context}`,
      );
    }

    throw error instanceof Error ? error : new InternalServerErrorException('Unknown error');
  }
}

