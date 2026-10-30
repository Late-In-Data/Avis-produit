"""Télécharge, audite et échantillonne uniformément les textes uniques SNAP."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from sentiment.data import (
    ROOT, SOURCE_URL, SOURCE_PAGE, SAMPLE_PATH, AUDIT_PATH,
    clean_text, iter_reviews, text_key, rating_label,
)


def prepare(raw: Path, sample_size: int, seed: int):
    rng = random.Random(seed)
    counts, missing = Counter(), Counter()
    # Un bit par classe : négatif=1, 3 étoiles=2, positif=4.
    seen: dict[str, int] = {}
    reservoir = []
    total = duplicate_rows = invalid_rows = unique_valid = 0
    minimum_time, maximum_time = None, None
    for record in iter_reviews(raw):
        total += 1
        for field in ("product/productId", "review/score", "review/time", "review/text"):
            if not record.get(field, "").strip():
                missing[field] += 1
        try:
            numeric_score = float(record.get("review/score", "nan"))
            score = int(numeric_score)
            timestamp = int(record["review/time"])
            text = clean_text(record.get("review/text", ""))
            if numeric_score != score or score not in range(1, 6) or not text or not record.get("product/productId"):
                raise ValueError("Invalid review")
            date = pd.Timestamp(timestamp, unit="s", tz="UTC").isoformat()
        except (ValueError, KeyError, OverflowError):
            invalid_rows += 1
            continue
        counts[score] += 1
        minimum_time = timestamp if minimum_time is None else min(minimum_time, timestamp)
        maximum_time = timestamp if maximum_time is None else max(maximum_time, timestamp)
        key = text_key(text)
        bit = 1 if score <= 2 else 2 if score == 3 else 4
        if key in seen:
            duplicate_rows += 1
            seen[key] |= bit
            continue
        seen[key] = bit
        unique_valid += 1
        row = dict(product_id=record["product/productId"], score=score,
                   date=date, summary=clean_text(record.get("review/summary", "")),
                   text=text, text_key=key, label=rating_label(score) or "3 étoiles (hors modèle)")
        if len(reservoir) < sample_size:
            reservoir.append(row)
        else:
            index = rng.randrange(unique_valid)
            if index < sample_size:
                reservoir[index] = row
    # Exclure tout texte associé à des notes positives ET négatives dans la source entière.
    conflicts = {key for key, mask in seen.items() if mask & 1 and mask & 4}
    frame = pd.DataFrame(reservoir)
    if frame.empty:
        raise ValueError("Aucun avis valide dans la source.")
    conflict_sample = int(frame.text_key.isin(conflicts).sum())
    frame = frame.loc[~frame.text_key.isin(conflicts)].reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame.date, utc=True)
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(SAMPLE_PATH, index=False)
    with raw.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    audit = dict(
        source=SOURCE_PAGE, download_url=SOURCE_URL,
        retrieved_at=datetime.now(timezone.utc).isoformat(), source_sha256=digest,
        rows_raw=total, rows_valid=sum(counts.values()), invalid_rows=invalid_rows,
        missing_required=dict(missing), ratings_raw={str(k): counts[k] for k in range(1, 6)},
        unique_normalized_texts=unique_valid, duplicate_text_rows=duplicate_rows,
        conflicting_binary_texts=len(conflicts), conflicts_removed_from_sample=conflict_sample,
        sample_requested=sample_size, sample_rows=len(frame), seed=seed,
        sample_ratings={str(k): int(v) for k, v in frame.score.value_counts().sort_index().items()},
        sample_binary_rows=int(frame.score.ne(3).sum()),
        sample_products=int(frame.product_id.nunique()),
        period_start=pd.Timestamp(minimum_time, unit="s", tz="UTC").isoformat(),
        period_end=pd.Timestamp(maximum_time, unit="s", tz="UTC").isoformat(),
        fields_available=["product_id", "score", "date", "summary", "text"],
        source_encoding="Latin-1 (archive historique non UTF-8)",
        aspect_labels_available=False,
        sampling="Réservoir uniforme sur les textes normalisés uniques, première occurrence conservée, puis retrait des conflits binaires.",
        labeling="1–2 étoiles = Négatif ; 4–5 = Positif ; 3 étoiles exclues du modèle, sans supposer leur neutralité.",
    )
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=True, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/finefoods.txt.gz")
    args = parser.parse_args()
    if args.sample_size < 100:
        parser.error("--sample-size doit être >= 100")
    if not args.raw.exists():
        args.raw.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.raw.with_suffix(".download")
        print("Téléchargement de la source SNAP (~130 Mo)…", flush=True)
        try:
            urllib.request.urlretrieve(SOURCE_URL, temporary)
            temporary.replace(args.raw)
        finally:
            temporary.unlink(missing_ok=True)
    prepare(args.raw, args.sample_size, args.seed)


if __name__ == "__main__":
    main()
