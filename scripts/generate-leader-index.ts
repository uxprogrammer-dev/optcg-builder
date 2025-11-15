import { readdir, readFile, writeFile } from 'fs/promises';
import { join } from 'path';

type RawCard = {
  id: string;
  name: string;
  type: string;
  color?: string | string[];
  family?: string | null;
  set?: { name?: string | null };
};

(async () => {
  const cardsDir = join(__dirname, '../data/cards/en');
  const files = await readdir(cardsDir);
  const leaders: Record<string, unknown> = {};

  for (const file of files) {
    if (!file.endsWith('.json')) continue;
    const filePath = join(cardsDir, file);
    try {
      const content = await readFile(filePath, 'utf8');
      const data = JSON.parse(content) as RawCard[];
      data
        .filter((card) => card.type?.toUpperCase() === 'LEADER')
        .forEach((card) => {
          const colors = Array.isArray(card.color)
            ? card.color
            : (card.color ?? '')
                .split(/[\/,]/)
                .map((part) => part.trim())
                .filter(Boolean);

          const subtypes = (card.family ?? '')
            .split(/[\/,]/)
            .map((part) => part.trim())
            .filter(Boolean);

          leaders[card.id.toUpperCase()] = {
            id: card.id,
            name: card.name,
            colors,
            subtypes,
            setName: card.set?.name ?? null,
            sourceFile: file,
          };
        });
    } catch (error) {
      console.warn(`Failed to process ${file}: ${(error as Error).message}`);
    }
  }

  const sorted = Object.keys(leaders)
    .sort()
    .reduce<Record<string, unknown>>((acc, key) => {
      acc[key] = leaders[key];
      return acc;
    }, {});

  const outPath = join(__dirname, '../data/leaders.json');
  await writeFile(outPath, JSON.stringify(sorted, null, 2));
  console.log(`Generated leader index with ${Object.keys(sorted).length} entries at ${outPath}`);
})();
