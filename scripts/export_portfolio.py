"""Exporte le modèle lexical pour une inférence identique sans serveur Python."""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
from sentiment.data import ROOT, MODEL_PATH


def main():
    target = ROOT / "portfolio/dist"
    target.mkdir(parents=True, exist_ok=True)
    model = joblib.load(MODEL_PATH)
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    assert list(model.classes_) == ["Négatif", "Positif"]
    assert vectorizer.stop_words is None and vectorizer.norm == "l2" and vectorizer.sublinear_tf
    features = {term: [float(vectorizer.idf_[index]), float(classifier.coef_[0, index])]
                for term, index in vectorizer.vocabulary_.items()}
    exported = dict(classes=list(model.classes_), intercept=float(classifier.intercept_[0]),
                    features=features, token_pattern=vectorizer.token_pattern,
                    ngram_range=list(vectorizer.ngram_range), sublinear_tf=True, norm="l2")
    (target / "model.json").write_text(json.dumps(exported, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    results = {}
    for key, filename in [("initial", "metrics.json"), ("robustness", "robustness.json"),
                          ("review", "assisted_review_summary.json"), ("transformer", "transformer_benchmark.json"),
                          ("comparison", "model_comparison.json")]:
        path = ROOT / "reports" / filename
        if path.exists():
            results[key] = json.loads(path.read_text(encoding="utf-8"))
    (target / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(features)} variables exportées ; {(target / 'model.json').stat().st_size/1024**2:.2f} Mo")


if __name__ == "__main__":
    main()
