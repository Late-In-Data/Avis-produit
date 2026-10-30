import json
import pandas as pd
import pytest
from sentiment.data import ROOT
import numpy as np
from scripts.compare_models import macro_f1


def test_macro_f1_bootstrap_statistic():
    assert macro_f1(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1])) == pytest.approx((2/3 + 4/5)/2)


@pytest.mark.skipif(not (ROOT / "reports/unseen_products_splits.csv").exists(), reason="Évaluations complémentaires requises")
def test_robustness_splits_respect_their_contracts():
    report = json.loads((ROOT / "reports/robustness.json").read_text(encoding="utf-8"))
    temporal = report["temporal"]["periods"]
    assert temporal["train"][1] < temporal["validation"][0]
    assert temporal["validation"][1] < temporal["test"][0]
    for name in ["temporal", "unseen_products"]:
        manifest = pd.read_csv(ROOT / f"reports/{name}_splits.csv")
        assert manifest.text_key.is_unique
        assert manifest.split.value_counts().to_dict() == report[name]["counts"]
    groups = pd.read_csv(ROOT / "reports/unseen_products_splits.csv")
    assert groups.groupby("product_id").split.nunique().max() == 1


@pytest.mark.skipif(not (ROOT / "reports/assisted_annotations.csv").exists(), reason="Annotations détaillées conservées localement")
def test_assisted_annotations_are_complete_and_honestly_labeled():
    annotations = pd.read_csv(ROOT / "reports/assisted_annotations.csv")
    assert len(annotations) == 80
    assert annotations.review_id.is_unique
    assert annotations.text_key.str.fullmatch(r"[0-9a-f]{64}").all()
    assert annotations.reason.notna().all()
    assert set(annotations.assisted_sentiment) <= {"Positif", "Négatif", "Mixte", "Indéterminé"}
    assert annotations.possible_rating_mismatch.sum() == 2


@pytest.mark.skipif(not (ROOT / "data/processed/transformer_predictions.parquet").exists(), reason="Benchmark requis")
def test_transformer_compares_the_same_test_cases():
    frame = pd.read_parquet(ROOT / "data/processed/transformer_predictions.parquet")
    original = pd.read_parquet(ROOT / "data/processed/test_predictions.parquet")
    assert frame.text_key.tolist() == original.text_key.tolist()
    assert frame.prediction.tolist() == original.prediction.tolist()
    assert frame.transformer_positive_score.between(0, 1).all()
    result = json.loads((ROOT / "reports/transformer_benchmark.json").read_text(encoding="utf-8"))
    assert sum(map(sum, result["transformer"]["metrics"]["confusion_matrix"])) == len(frame)
