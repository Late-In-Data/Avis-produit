"""Reconstruire le pipeline lexical évalué à partir de ses poids JSON suivis par Git."""
import json
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def restore_frozen_model(export_path: Path, model_path: Path):
    source = json.loads(export_path.read_text(encoding="utf-8"))
    terms = list(source["features"])
    vectorizer = TfidfVectorizer(vocabulary={term: index for index, term in enumerate(terms)},
                                 token_pattern=source["token_pattern"], ngram_range=tuple(source["ngram_range"]),
                                 sublinear_tf=source["sublinear_tf"], norm=source["norm"],
                                 min_df=3, max_df=.98, max_features=60000)
    vectorizer.idf_ = np.array([source["features"][term][0] for term in terms])
    vectorizer._tfidf.n_features_in_ = len(terms)
    classifier = LogisticRegression(C=2, class_weight="balanced", max_iter=500, random_state=42)
    classifier.classes_ = np.array(source["classes"])
    classifier.coef_ = np.array([[source["features"][term][1] for term in terms]])
    classifier.intercept_ = np.array([source["intercept"]])
    classifier.n_features_in_ = len(terms)
    pipeline = Pipeline([("tfidf", vectorizer), ("classifier", classifier)])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = model_path.with_name(model_path.name+"."+uuid4().hex+".tmp")
    try:
        joblib.dump(pipeline, temporary)
        temporary.replace(model_path)
    finally:
        temporary.unlink(missing_ok=True)
    return pipeline
