import gzip
import json

import joblib
import pandas as pd
import pytest

from sentiment.data import ROOT, SAMPLE_PATH, MODEL_PATH, METRICS_PATH, clean_text, iter_reviews, rating_label, safe_csv, text_key


def test_label_policy_and_deduplication():
    assert [rating_label(i) for i in range(1, 6)] == ["Négatif", "Négatif", None, "Positif", "Positif"]
    assert text_key("Good<br /> FOOD &amp; drink") == text_key("good food & drink")
    assert clean_text("<b>Not good</b>") == "Not good"


def test_snap_reader_preserves_colons_and_last_record(tmp_path):
    path = tmp_path / "reviews.gz"
    with gzip.open(path, "wt", encoding="latin-1") as stream:
        stream.write("review/text: Taste: café\nreview/score: 2.0\n\nreview/text: Good\n")
    rows = list(iter_reviews(path))
    assert len(rows) == 2
    assert rows[0]["review/text"] == "Taste: café"
    assert rows[1]["review/text"] == "Good"


def test_export_neutralizes_formula_cells():
    frame = pd.DataFrame({"text": ["=1+1", " @SUM(A1)", "Ordinary review"]})
    exported = pd.read_csv(__import__("io").BytesIO(safe_csv(frame)))
    assert exported.text.tolist() == ["'=1+1", "' @SUM(A1)", "Ordinary review"]


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Préparer puis entraîner pour vérifier les artefacts réels")
def test_real_artifacts_have_disjoint_splits_and_consistent_metrics():
    sample = pd.read_parquet(SAMPLE_PATH)
    splits = pd.read_csv(ROOT / "reports/splits.csv")
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    assert sample.text_key.is_unique
    assert splits.text_key.is_unique
    assert set(splits.text_key) == set(sample.loc[sample.score.ne(3), "text_key"])
    assert splits.split.value_counts().to_dict() == metrics["split_counts"]
    test = pd.read_parquet(ROOT / "data/processed/test_predictions.parquet")
    model = joblib.load(MODEL_PATH)
    assert list(model.predict(test.text)) == test.prediction.tolist()
    assert (test.label == test.prediction).mean() == metrics["test"]["accuracy"]
    assert sum(map(sum, metrics["test"]["confusion_matrix"])) == len(test)
    assert len(model.named_steps["tfidf"].vocabulary_) <= 60000
    assert list(model.classes_) == ["Négatif", "Positif"]
