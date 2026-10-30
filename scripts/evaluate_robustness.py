"""Évaluations complémentaires à hyperparamètres figés, sans modifier la V1."""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from scripts.train import evaluate
from sentiment.data import ROOT, SAMPLE_PATH


def fresh_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=.98,
                                  max_features=60000, sublinear_tf=True)),
        ("classifier", LogisticRegression(C=2, class_weight="balanced", max_iter=500, random_state=42)),
    ])


def evaluate_split(name, train, validation, test):
    assert set(train.text_key).isdisjoint(test.text_key)
    assert set(validation.text_key).isdisjoint(test.text_key)
    assert set(train.text_key).isdisjoint(validation.text_key)
    model = fresh_pipeline().fit(train.text, train.label)
    predicted = model.predict(test.text)
    baseline = DummyClassifier(strategy="most_frequent").fit(train.text, train.label)
    result = dict(test=evaluate(test.label, predicted),
                  baseline=evaluate(test.label, baseline.predict(test.text)),
                  counts={"train": len(train), "validation": len(validation), "test": len(test)},
                  classes={split: subset.label.value_counts().to_dict() for split, subset in
                           [("train", train), ("validation", validation), ("test", test)]},
                  periods={split: [subset.date.min().isoformat(), subset.date.max().isoformat()]
                           for split, subset in [("train", train), ("validation", validation), ("test", test)]},
                  overlap_products_train_test=len(set(train.product_id) & set(test.product_id)),
                  protocol="Pipeline et hyperparamètres V1 figés ; ajustement sur train uniquement. Validation réservée, non utilisée pour sélectionner ou régler le modèle.")
    predictions = test.assign(prediction=predicted)
    predictions.to_parquet(ROOT / f"data/processed/{name}_predictions.parquet", index=False)
    manifest = pd.concat([part[["text_key", "product_id"]].assign(split=split)
                          for split, part in [("train", train), ("validation", validation), ("test", test)]])
    manifest.to_csv(ROOT / f"reports/{name}_splits.csv", index=False)
    print(name, json.dumps(result, ensure_ascii=True), flush=True)
    return result


def main():
    frame = pd.read_parquet(SAMPLE_PATH)
    frame = frame.loc[frame.score.ne(3)].copy()
    ordered = frame.sort_values(["date", "text_key"])
    # Coupures par journées entières : aucune date à cheval entre les ensembles.
    cutoff1 = ordered.iloc[int(.70 * len(ordered))].date
    cutoff2 = ordered.iloc[int(.85 * len(ordered))].date
    train = ordered.loc[ordered.date.lt(cutoff1)]
    val = ordered.loc[ordered.date.ge(cutoff1) & ordered.date.lt(cutoff2)]
    test = ordered.loc[ordered.date.ge(cutoff2)]
    temporal = evaluate_split("temporal", train, val, test)
    assert train.date.max() < val.date.min() <= val.date.max() < test.date.min()
    group_split = GroupShuffleSplit(n_splits=1, test_size=.15, random_state=42)
    dev_ids, test_ids = next(group_split.split(frame, groups=frame.product_id))
    dev, test = frame.iloc[dev_ids], frame.iloc[test_ids]
    train_ids, val_ids = next(GroupShuffleSplit(n_splits=1, test_size=.15/.85, random_state=43).split(dev, groups=dev.product_id))
    train, val = dev.iloc[train_ids], dev.iloc[val_ids]
    assert set(train.product_id).isdisjoint(test.product_id)
    assert set(val.product_id).isdisjoint(test.product_id)
    assert set(train.product_id).isdisjoint(val.product_id)
    product = evaluate_split("unseen_products", train, val, test)
    report = dict(temporal=temporal, unseen_products=product,
                  limitations=["Deux stress tests séparés : pas de garantie simultanée sur produits inconnus et période future.",
                               "Un produit est identifié par l'ASIN conservé lors de la déduplication ; variantes et familles ne sont pas regroupées.",
                               "Les dates restent dans le corpus historique, jusqu'en 2012.",
                               "Les trois tests portent sur des populations différentes ; les écarts ne sont pas des effets causaux."])
    (ROOT / "reports/robustness.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
