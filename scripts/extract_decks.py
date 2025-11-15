
import argparse
import json
from urllib.parse import urlparse, parse_qs, unquote
from collections import Counter

def parse_deck_url(url: str):
    """
    Parse a single deckgen URL into a structured dict:
      - Metadata: deck_name, author, date, country, tournament, placement, host
      - Decklist: dict mapping card_code -> count
    """
    parsed = urlparse(url.strip())
    if not parsed.query:
        return None  # skip if not a deckgen URL or no query present

    qs = parse_qs(parsed.query)

    def get_one(key, default=""):
        val = qs.get(key, [""])
        return unquote(val[0]) if val else default

    deck_name = get_one("dn")
    author = get_one("au")
    date = get_one("date")
    country = get_one("cn")
    tn = get_one("tn")
    pl = get_one("pl")
    host = get_one("hs")

    tournament = tn
    placement = pl

    # Parse decklist from "dg" param (e.g., '1nOP13-079a3nOP05-082a...')
    decklist_str = get_one("dg")
    decklist = {}
    if decklist_str:
        parts = [p for p in decklist_str.split("a") if p]
        for part in parts:
            if "n" in part:
                cnt_str, code = part.split("n", 1)
                try:
                    cnt = int(cnt_str)
                except ValueError:
                    continue
                code = code.strip()
                if code:
                    decklist[code] = decklist.get(code, 0) + cnt

    record = {
        "deck_name": deck_name,
        "author": author,
        "date": date,
        "country": country,
        "tournament": tournament,
        "placement": placement,
        "host": host,
        "decklist": decklist
    }
    if not any([deck_name, author, date, country, tournament, placement, host, decklist]):
        return None
    return record


def main():
    ap = argparse.ArgumentParser(description="Extract deck metadata + decklists from deckgen links in a text file.")
    ap.add_argument("--input", required=True, help="Path to a text file with one deckgen URL per line.")
    ap.add_argument("--output", required=True, help="Output JSON filename (e.g., op13_decks.json).")
    ap.add_argument("--dup-key", default="deck_name,author,date",
                    help="Comma-separated fields to define duplicates (default: deck_name,author,date).")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    records = []
    for ln in lines:
        rec = parse_deck_url(ln)
        if rec:
            records.append(rec)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    key_fields = [k.strip() for k in args.dup_key.split(",") if k.strip()]
    def make_key(rec):
        return tuple(rec.get(field, "") for field in key_fields)

    from collections import Counter
    counts = Counter(make_key(r) for r in records)
    dups = [k for k, c in counts.items() if c > 1]

    print(f"Total records: {len(records)}")
    if dups:
        print(f"Duplicates found ({len(dups)} unique keys):")
        for k in dups:
            print(" - " + " | ".join(f"{field}={val}" for field, val in zip(key_fields, k)))
    else:
        print("No duplicates found.")

if __name__ == "__main__":
    main()
