import { Injectable, Logger } from '@nestjs/common';
import { OptcgCard } from '@shared/types/optcg-card';
import { readdir, readFile } from 'fs/promises';
import { join } from 'path';

type RawLocalCard = {
  id: string;
  code?: string;
  name: string;
  ability?: string | null;
  text?: string | null;
  type: string;
  color?: string | null;
  cost?: number | string | null;
  power?: number | string | null;
  life?: number | string | null;
  family?: string | null;
  counter?: number | string | null;
  rarity?: string | null;
  set?: {
    name?: string | null;
  };
  attribute?: {
    name?: string | null;
  };
  images?: {
    large?: string | null;
  };
  market_price?: number | null;
  inventory_price?: number | null;
};

@Injectable()
export class CardDataRepository {
  private readonly logger = new Logger(CardDataRepository.name);
  private readonly cache = new Map<string, OptcgCard>();
  private isLoaded = false;

  async findById(cardSetId: string): Promise<OptcgCard | null> {
    if (!this.isLoaded) {
      await this.loadLocalData().catch((error) => {
        this.logger.warn(`Failed to load local card data: ${(error as Error).message}`);
      });
    }

    return this.cache.get(cardSetId.toUpperCase()) ?? null;
  }

  async getAllCards(): Promise<OptcgCard[]> {
    if (!this.isLoaded) {
      await this.loadLocalData().catch((error) => {
        this.logger.warn(`Failed to load local card data: ${(error as Error).message}`);
      });
    }

    return Array.from(this.cache.values());
  }

  cacheCard(card: OptcgCard): void {
    this.cache.set(card.id.toUpperCase(), card);
  }

  private async loadLocalData(): Promise<void> {
    if (this.isLoaded) {
      return;
    }

    try {
      const dataDir = join(__dirname, '../../../data/cards/en');
      const files = await readdir(dataDir);

      for (const file of files.filter((name) => name.endsWith('.json'))) {
        const filePath = join(dataDir, file);
        try {
          const content = await readFile(filePath, 'utf8');
          const rawCards = JSON.parse(content) as RawLocalCard[];
          for (const raw of rawCards) {
            const card = this.mapLocalCard(raw);
            if (card) {
              this.cache.set(card.id.toUpperCase(), card);
            }
          }
        } catch (error) {
          this.logger.warn(`Failed to load ${file}: ${(error as Error).message}`);
        }
      }

      this.isLoaded = true;
      this.logger.log(`Loaded ${this.cache.size} cards from local data`);
    } catch (error) {
      this.logger.error(`Unable to load local card data: ${(error as Error).message}`);
    }
  }

  private mapLocalCard(raw: RawLocalCard): OptcgCard | null {
    if (!raw.id || !raw.name) {
      return null;
    }

    const text = raw.text ?? raw.ability ?? null;
    const subtypes =
      raw.family?.split(/[\/,]/).map((part) => part.trim()).filter(Boolean) ?? [];

    return {
      id: raw.id,
      name: raw.name,
      text,
      type: raw.type,
      color: raw.color ?? null,
      cost:
        raw.cost === undefined || raw.cost === null || raw.cost === '' ? null : String(raw.cost),
      power:
        raw.power === undefined || raw.power === null || raw.power === ''
          ? null
          : String(raw.power),
      life:
        raw.life === undefined || raw.life === null || raw.life === ''
          ? null
          : String(raw.life),
      attribute: raw.attribute?.name ?? null,
      subtypes,
      rarity: raw.rarity ?? null,
      setId: raw.code ?? raw.id.split('-')[0] ?? 'UNKNOWN',
      setName: raw.set?.name ?? 'Local Data',
      imageUrl: raw.images?.large ?? '',
      marketPrice: raw.market_price ?? null,
      inventoryPrice: raw.inventory_price ?? null,
      raw: {
        inventory_price: raw.inventory_price ?? null,
        market_price: raw.market_price ?? null,
        card_name: raw.name,
        set_name: raw.set?.name ?? 'Local Data',
        card_text: text ?? '',
        set_id: raw.code ?? raw.id.split('-')[0] ?? 'UNKNOWN',
        rarity: raw.rarity ?? '',
        card_set_id: raw.id,
        card_color: raw.color ?? '',
        card_type: raw.type,
        life: raw.life === undefined || raw.life === null ? 'NULL' : String(raw.life),
        card_cost:
          raw.cost === undefined || raw.cost === null || raw.cost === '' ? 'NULL' : String(raw.cost),
        card_power:
          raw.power === undefined || raw.power === null || raw.power === ''
            ? 'NULL'
            : String(raw.power),
        sub_types: subtypes.join(' '),
        counter_amount: toNumber(raw.counter),
        attribute: raw.attribute?.name ?? null,
        date_scraped: null,
        card_image_id: raw.id,
        card_image: raw.images?.large ?? '',
      },
    };
  }
}

function toNumber(value: unknown): number {
  if (value === undefined || value === null || value === '') {
    return 0;
  }
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : 0;
}

