"""Thèmes exploratoires globaux, distincts du sentiment et de l'analyse par aspect."""
from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from threadpoolctl import threadpool_limits

from sentiment.data import ROOT
from sentiment.storage import write_json, write_parquet

TOPIC_MODEL = ROOT / "models/topics.joblib"
TOPIC_REPORT = ROOT / "reports/topics.json"
TOPIC_ASSIGNMENTS = ROOT / "data/processed/topic_assignments.parquet"
GENERIC_WORDS = {"like", "good", "great", "really", "just", "don", "ve", "ll", "did", "does",
                 "didn", "doesn", "didnt", "doesnt", "isn", "wasn", "love", "loved", "recommend",
                 "excellent", "best", "product", "products", "buy", "bought", "try", "tried", "use",
                 "used", "using", "make", "makes", "making"}


def corpus_fingerprint(frame):
    return hashlib.sha256("\n".join(frame.text_key).encode()).hexdigest()


def assign_topics(bundle, texts):
    matrix = bundle["vectorizer"].transform(texts)
    with threadpool_limits(limits=4):
        weights = bundle["nmf"].transform(matrix)
    total = weights.sum(axis=1)
    dominant = weights.argmax(axis=1).astype(int)
    dominant[total <= 1e-12] = -1
    share = np.divide(weights.max(axis=1), total, out=np.zeros(len(total)), where=total>1e-12)
    return dominant, share


def build_topics(frame, n_topics=8):
    if not 2 <= n_topics <= 20:
        raise ValueError("Choisir entre 2 et 20 thèmes.")
    vectorizer = TfidfVectorizer(stop_words=sorted(set(ENGLISH_STOP_WORDS) | GENERIC_WORDS), min_df=10, max_df=.7,
                                 max_features=8000, ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform(frame.text)
    if min(matrix.shape) < n_topics:
        raise ValueError("Corpus trop petit pour ce nombre de thèmes.")
    nmf = NMF(n_components=n_topics, init="nndsvda", random_state=42, max_iter=400)
    with threadpool_limits(limits=4):
        weights = nmf.fit_transform(matrix)
    terms = vectorizer.get_feature_names_out()
    keywords = [[str(terms[i]) for i in component.argsort()[::-1][:10]] for component in nmf.components_]
    names = [f"Thème {i+1} · " + " / ".join(words[:3]) for i, words in enumerate(keywords)]
    bundle = dict(vectorizer=vectorizer, nmf=nmf, names=names,
                  corpus_sha256=corpus_fingerprint(frame))
    totals = weights.sum(axis=1)
    dominant = weights.argmax(axis=1).astype(int)
    dominant[totals <= 1e-12] = -1
    strength = np.divide(weights.max(axis=1), totals, out=np.zeros(len(totals)), where=totals>1e-12)
    assignments = frame[["text_key"]].assign(topic_id=dominant, topic_strength=strength)
    topics = []
    for index, words in enumerate(keywords):
        selected = frame.loc[dominant == index]
        binary = selected.loc[selected.score.ne(3)]
        topics.append(dict(id=index, label=names[index], keywords=words, documents=len(selected),
                           binary_documents=len(binary), positive=int(binary.score.ge(4).sum()),
                           negative=int(binary.score.le(2).sum()), three_stars=int(selected.score.eq(3).sum())))
    report = dict(method="TF-IDF + NMF", n_topics=n_topics, seed=42,
                  corpus_sha256=bundle["corpus_sha256"], source="Amazon Fine Foods — échantillon dédupliqué",
                  documents=len(frame), unassigned=int((dominant < 0).sum()),
                  topics=topics, n_iter=int(nmf.n_iter_), converged=bool(nmf.n_iter_ < nmf.max_iter),
                  reconstruction_error=float(nmf.reconstruction_err_),
                  parameters=dict(min_df=10, max_df=.7, max_features=8000, ngram_range=[1, 2],
                                  stop_words="anglais + mots génériques d'opinion et d'achat"),
                  limitations=["Thèmes exploratoires ajustés sur tout l'échantillon, y compris les trois étoiles ; aucune modification du classificateur de sentiment.",
                               "Intitulés automatiques issus des mots dominants, sans validation humaine de la cohérence des thèmes.",
                               "Un texte peut évoquer plusieurs thèmes ; le décompte utilise seulement la composante dominante et n'est pas une probabilité.",
                               "Ces thèmes globaux ne sont pas des sentiments par aspect.",
                               "Les modèles de thèmes et de sentiment ne sont pas entraînés sur Reddit ; leur application éventuelle à Reddit est hors domaine évalué."])
    return bundle, assignments, report


def save_topics(bundle, assignments, report):
    TOPIC_MODEL.parent.mkdir(parents=True, exist_ok=True)
    temporary = TOPIC_MODEL.with_name(TOPIC_MODEL.name + "." + uuid4().hex + ".tmp")
    try:
        joblib.dump(bundle, temporary)
        temporary.replace(TOPIC_MODEL)
    finally:
        temporary.unlink(missing_ok=True)
    write_parquet(TOPIC_ASSIGNMENTS, assignments)
    write_json(TOPIC_REPORT, report)
