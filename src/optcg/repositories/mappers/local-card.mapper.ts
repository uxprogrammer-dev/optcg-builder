import { OptcgCard, OptcgCardApiResponse } from '@shared/types/optcg-card';

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

export function mapLocalCard(raw: RawLocalCard): OptcgCard | null {
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
    cost: toStringOrNull(raw.cost),
    power: toStringOrNull(raw.power),
    life: toStringOrNull(raw.life),
    attribute: raw.attribute?.name ?? null,
    subtypes,
    rarity: raw.rarity ?? null,
    setId: raw.code ?? raw.id.split('-')[0] ?? 'UNKNOWN',
    setName: raw.set?.name ?? 'Local Data',
    imageUrl: raw.images?.large ?? '',
    marketPrice: raw.market_price ?? null,
    inventoryPrice: raw.inventory_price ?? null,
    raw: toApiResponse(raw, text, subtypes),
  };
}

function toStringOrNull(value: unknown): string | null {
  if (value === undefined || value === null || value === '') {
    return null;
  }
  return String(value);
}

function toApiResponse(
  raw: RawLocalCard,
  text: string | null,
  subtypes: string[],
): OptcgCardApiResponse {
  return {
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
    life: toStringOrNull(raw.life) ?? 'NULL',
    card_cost: toStringOrNull(raw.cost) ?? 'NULL',
    card_power: toStringOrNull(raw.power) ?? 'NULL',
    sub_types: subtypes.join(' '),
    counter_amount: 0,
    attribute: raw.attribute?.name ?? null,
    date_scraped: null,
    card_image_id: raw.id,
    card_image: raw.images?.large ?? '',
  };
}

