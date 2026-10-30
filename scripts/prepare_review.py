"""Tire 80 avis, 20 dans chaque cellule de confusion, pour une revue qualitative."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from sentiment.data import ROOT


def main():
    frame = pd.read_parquet(ROOT / "data/processed/test_predictions.parquet")
    rows = frame.groupby(["label", "prediction"], group_keys=False).sample(n=20, random_state=2026)
    # Ordre mélangé. La revue assistée reste non aveugle aux notes et prédictions.
    rows = rows.sample(frac=1, random_state=2027).reset_index(drop=True)
    rows.insert(0, "review_id", [f"R{i:03d}" for i in range(1, len(rows)+1)])
    (ROOT / "reports/review_sample.json").write_text(rows[["review_id", "text_key", "score", "label", "prediction", "text"]].to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    rows[["review_id", "text_key", "text"]].assign(human_sentiment="", human_reason="", annotator="").to_csv(ROOT / "reports/human_review_template.csv", index=False, encoding="utf-8-sig")
    print(f"{len(rows)} avis ; 20 par cellule de confusion. Échantillon volontairement enrichi en erreurs, non représentatif du test.")


if __name__ == "__main__":
    main()
