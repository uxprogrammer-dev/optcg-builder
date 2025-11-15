# Card & Deck Schema Overview

This note captures the agreed-upon representation to support prompt→deck training.

## Card Corpus Snapshot

The script `python -m ml.data.profiling` aggregates the JSON exports under `data/cards/en`.

- Unique cards: derived from the `id` field across all sets, stored in `ml/artifacts/card_profile_en.json`.
- Unique leaders: counted via the `type == "LEADER"` marker.
- Dimensions profiled: color, type, attribute, family, set, cost, and power distributions.
- Output refresh: rerun the script after every data update to keep the profile synced.

## Normalised Card Record

Implemented via `ml/data/card_schema.py::CardRecord`, retaining:

- Core identifiers: `id`, `code`, `name`, `type`, `color`, `rarity`.
- Gameplay stats: `cost`, `power`, `counter`, `ability`, `trigger`, `attribute`, `family`.
- Metadata: `set_name`, plus the untouched `raw` payload for forward compatibility.

## Deck Schema

Represented by `DeckSchema` with the following invariants:

- One `leader_id`.
- `main_deck` of exactly 50 card identifiers (canonical IDs, duplicates allowed for now).
- Optional `sideboard` extension (future proofing).
- Convenience helpers `as_sequence()` (leader + deck flattened) and `validate()`.

Reserved tokens are centralised in `ml/config/schemas.py::DeckConfig`.

## Prompt→Deck Example

Training instances use `PromptDeckExample`:

- `prompt`: natural-language goal supplied by the user or generator.
- `deck`: `DeckSchema` object.
- Optional context: `prompt_style`, `quality_tags`, `split`.
- Serialisation helper `to_record()` feeds JSONL/TFRecord writers.

Prompt configuration defaults (`PromptConfig`) cap prompts at 256 wordpiece tokens with a vocabulary limit of 16k.

## Training Split Definitions

`TrainingSplits` specifies the default 80/10/10 split. The helper `normalised()` ensures future adjustments still sum to 1.0.

---

These schema definitions provide consistency for the downstream synthetic dataset builder, feature pipelines, and Keras model training stages.

