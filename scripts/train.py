"""Compare sur validation, évalue une seule fois sur test et sauvegarde le modèle."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sentiment.data import SAMPLE_PATH, MODEL_PATH, METRICS_PATH, ROOT

LABELS = ["Négatif", "Positif"]


def evaluate(y, predicted):
    return dict(accuracy=accuracy_score(y, predicted),
                macro_f1=f1_score(y, predicted, average="macro"),
                report=classification_report(y, predicted, labels=LABELS, output_dict=True, zero_division=0),
                confusion_matrix=confusion_matrix(y, predicted, labels=LABELS).tolist())


def main():
    start = time.perf_counter()
    frame = pd.read_parquet(SAMPLE_PATH)
    binary = frame.loc[frame.score.ne(3)].copy()
    train, rest = train_test_split(binary, test_size=.30, stratify=binary.label, random_state=42)
    validation, test = train_test_split(rest, test_size=.50, stratify=rest.label, random_state=42)
    assert not set(train.text_key) & set(test.text_key)
    assert not set(train.text_key) & set(validation.text_key)
    assert not set(validation.text_key) & set(test.text_key)
    candidates = {
        "Régression logistique": LogisticRegression(C=2, class_weight="balanced", max_iter=500, random_state=42),
        "Naive Bayes": MultinomialNB(alpha=.5),
    }
    scores, pipelines = {}, {}
    for name, estimator in candidates.items():
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=3,
                                      max_df=.98, max_features=60000, sublinear_tf=True)),
            ("classifier", estimator),
        ])
        pipeline.fit(train.text, train.label)
        scores[name] = evaluate(validation.label, pipeline.predict(validation.text))
        pipelines[name] = pipeline
        print(f"Validation {name}: macro-F1={scores[name]['macro_f1']:.4f}", flush=True)
    selected = max(scores, key=lambda name: scores[name]["macro_f1"])
    model = pipelines[selected]
    # Le modèle sauvegardé est exactement celui évalué, sans réentraînement sur le test.
    predictions = model.predict(test.text)
    dummy = DummyClassifier(strategy="most_frequent").fit(train.text, train.label)
    baseline = evaluate(test.label, dummy.predict(test.text))
    metrics = dict(selected_model=selected, selection_metric="macro-F1 sur validation",
                   validation=scores, test=evaluate(test.label, predictions), baseline=baseline,
                   labels=LABELS, seed=42, split_counts={"train": len(train), "validation": len(validation), "test": len(test)},
                   split_class_counts={name: subset.label.value_counts().to_dict()
                                       for name, subset in [("train", train), ("validation", validation), ("test", test)]},
                   sklearn_version=sklearn.__version__, elapsed_seconds=round(time.perf_counter()-start, 2),
                   features="Texte du corps uniquement ; aucune note, date, identité ou fiche produit.",
                   limitations=["Labels déduits des notes, sans annotation humaine du sentiment.",
                                "Textes normalisés uniques ; les paraphrases et auteurs communs peuvent subsister.",
                                "Split aléatoire stratifié : ne mesure pas un déploiement futur ni des produits jamais vus.",
                                "Avis alimentaires historiques en anglais ; pas de validation en français ni par aspect."])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    errors = test.copy()
    errors["prediction"] = predictions
    errors["split"] = "test"
    errors.to_parquet(ROOT / "data/processed/test_predictions.parquet", index=False)
    split_manifest = pd.concat([subset[["text_key"]].assign(split=name)
                               for name, subset in [("train", train), ("validation", validation), ("test", test)]])
    split_manifest.to_csv(ROOT / "reports/splits.csv", index=False)
    print(json.dumps({"model": selected, "test": metrics["test"], "baseline": baseline,
                      "seconds": metrics["elapsed_seconds"]}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
