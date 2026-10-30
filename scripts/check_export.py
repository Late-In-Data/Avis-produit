"""Vérifie la parité Python/JavaScript sur tous les textes du test initial."""
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from sentiment.data import ROOT, MODEL_PATH


def main():
    frame = pd.read_parquet(ROOT / "data/processed/test_predictions.parquet")
    texts = frame.text.tolist() + ["Great great coffee, not bitter!", "Café très bon 123_test", "", "unknownwordqzzt"]
    result = subprocess.run(["node", str(ROOT / "scripts/check_browser_model.mjs")],
                            input=json.dumps(texts, ensure_ascii=True), text=True,
                            encoding="utf-8", capture_output=True, check=True, timeout=120)
    exported = json.loads(result.stdout)
    pipeline = joblib.load(MODEL_PATH)
    probabilities = pipeline.predict_proba(texts)[:, list(pipeline.classes_).index("Positif")]
    error = float(np.max(np.abs(probabilities - [row["positive"] for row in exported])))
    mismatches = sum(a != b["label"] for a, b in zip(pipeline.predict(texts), exported))
    assert error < 1e-10, error
    assert mismatches == 0
    report = dict(texts_checked=len(texts), label_mismatches=mismatches,
                  maximum_absolute_probability_difference=error,
                  scope="Tous les textes nettoyés du test initial et quatre cas de bord ; même pipeline exporté, sans réentraînement.")
    (ROOT / "reports/browser_parity.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))


if __name__ == "__main__":
    main()
