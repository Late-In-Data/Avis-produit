"""Intervalle bootstrap apparié sur l'écart de macro-F1 du même test."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sentiment.data import ROOT


def macro_f1(y, predicted):
    matrix = np.bincount(y*2+predicted, minlength=4).reshape(2, 2)
    diagonal = matrix.diagonal()
    denominator = matrix.sum(axis=0)+matrix.sum(axis=1)
    return float(np.divide(2*diagonal, denominator, out=np.zeros(2, dtype=float), where=denominator!=0).mean())


def main():
    data = pd.read_parquet(ROOT / "data/processed/transformer_predictions.parquet")
    y = data.label.eq("Positif").to_numpy(dtype=int)
    lexical = data.prediction.eq("Positif").to_numpy(dtype=int)
    transformer = data.transformer_prediction.eq("Positif").to_numpy(dtype=int)
    rng = np.random.default_rng(2026)
    differences = []
    for _ in range(2000):
        sample = rng.integers(0, len(data), len(data))
        differences.append(macro_f1(y[sample], transformer[sample])-macro_f1(y[sample], lexical[sample]))
    result = dict(difference_transformer_minus_lexical=macro_f1(y, transformer)-macro_f1(y, lexical),
                  percentile_interval_95=np.quantile(differences, [.025, .975]).tolist(),
                  bootstrap_repetitions=2000, seed=2026, test_rows=len(data),
                  limitations="Bootstrap apparié par avis sur le test partagé ; suppose des avis indépendants. Ne corrige pas les dépendances entre produits/auteurs ni les biais de labels ou de corpus.")
    (ROOT / "reports/model_comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
