export interface OptcgCardApiResponse {
  inventory_price: number | null;
  market_price: number | null;
  card_name: string;
  set_name: string;
  card_text: string | null;
  set_id: string;
  rarity: string | null;
  card_set_id: string;
  card_color: string | null;
  card_type: string;
  life: string | null;
  card_cost: string | null;
  card_power: string | null;
  sub_types: string | null;
  counter_amount: number;
  attribute: string | null;
  date_scraped: string | null;
  card_image_id: string;
  card_image: string;
}

export interface OptcgCard {
  id: string;
  name: string;
  text: string | null;
  type: string;
  color: string | null;
  cost: string | null;
  power: string | null;
  life: string | null;
  attribute: string | null;
  subtypes: string[];
  rarity: string | null;
  setId: string;
  setName: string;
  imageUrl: string;
  marketPrice: number | null;
  inventoryPrice: number | null;
  raw: OptcgCardApiResponse;
}

