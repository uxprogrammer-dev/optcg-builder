# OPTCG Deck Builder

Generate One Piece Trading Card Game leader suggestions and full deck lists from natural language prompts. The backend orchestrates OpenAI responses and the public OPTCG API, while the frontend (Next.js) renders card images and lets users download an image archive of the generated deck.

Card data and images are sourced from the public API documented at [optcgapi.com](https://optcgapi.com/documentation).

## Features

- Prompt-to-leader suggestions with strategic rationales (OpenAI GPT model).
- Deck generation that enriches each card with live data and official artwork.
- Downloadable ZIP archive bundling leader and deck card images.
- Next.js SPA with React Query for smooth request/response handling.

## Requirements

- Node.js 18+
- npm 9+
- An OpenAI API key with access to a GPT-4.1/GPT-4o family model.

## Getting Started

```bash
npm install
cp env.example .env
# populate .env with your OpenAI key and optional overrides

# Development: NestJS (http://localhost:3000) + Next.js (http://localhost:3001)
npm run dev
```

- The Next app proxies API calls to `/api` by default. Adjust `NEXT_PUBLIC_API_BASE_URL` in `.env` if you deploy the frontend separately.
- NestJS exposes all API routes under `/api/*`.

## Production Build

```bash
# Build NestJS (dist/) and export the Next app (apps/web/out/)
npm run build

# Ensure apps/web/out exists before starting the API server
npm start
```

`AppModule` automatically serves the exported Next bundle from `apps/web/out` when it exists; otherwise, only the API is served.

## API Overview

| Method | Endpoint                     | Description                                     |
|--------|------------------------------|-------------------------------------------------|
| POST   | `/api/ai/leaders`            | Suggest leaders for the provided prompt.        |
| POST   | `/api/decks`                 | Build a deck for the chosen leader and prompt.  |
| GET    | `/api/decks/:deckId`         | Retrieve cached deck details.                   |
| GET    | `/api/decks/:deckId/download`| Download leader + card images as a ZIP archive. |

All OPTCG card lookups use officially documented endpoints such as `/api/sets/card/{card_id}/` [^optcg].

## Frontend Flow

1. Enter a prompt describing colors, archetypes, or characters.
2. Review AI-suggested leader cards (with art) and select one.
3. Receive a generated deck list with images and strategic notes.
4. Download all images as a single ZIP archive for offline reference or printing.

## Folder Structure

- `src/` — NestJS application (AI, decks, files, OPTCG integrations).
- `libs/shared/` — Shared TypeScript contracts used by both backend and frontend.
- `apps/web/` — Next.js SPA (App Router, Tailwind, React Query).

## Deployment Notes

- The deck registry keeps results in-memory for one hour. Persist to a database or cache for multi-instance deployments.
- Respect the OPTCG API usage guidelines; the service is rate-limited and community operated.

[^optcg]: OPTCG API reference: https://optcgapi.com/documentation

