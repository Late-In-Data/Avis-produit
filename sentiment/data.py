"""Préparation reproductible. Aucune note n'entre dans les variables du modèle."""
from __future__ import annotations

import gzip
import hashlib
import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://snap.stanford.edu/data/finefoods.txt.gz"
SOURCE_PAGE = "https://snap.stanford.edu/data/web-FineFoods.html"
SAMPLE_PATH = ROOT / "data/processed/reviews.parquet"
AUDIT_PATH = ROOT / "reports/data_audit.json"
MODEL_PATH = ROOT / "models/sentiment.joblib"
METRICS_PATH = ROOT / "reports/metrics.json"


def clean_text(text: str) -> str:
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def text_key(text: str) -> str:
    return hashlib.sha256(clean_text(text).casefold().encode("utf-8")).hexdigest()


def rating_label(score: int) -> str | None:
    if score in (1, 2):
        return "Négatif"
    if score in (4, 5):
        return "Positif"
    return None


def iter_reviews(path: Path):
    """Lit les blocs SNAP sans charger l'archive en mémoire."""
    record = {}
    # L'archive historique SNAP contient des octets Latin-1 non valides en UTF-8.
    with gzip.open(path, "rt", encoding="latin-1", errors="strict") as stream:
        for line in stream:
            line = line.rstrip("\r\n")
            if not line:
                if record:
                    yield record
                    record = {}
                continue
            key, separator, value = line.partition(": ")
            if separator:
                record[key] = value
    if record:
        yield record


def safe_csv(frame) -> bytes:
    """Neutralise les formules lors de l'ouverture d'un export dans Excel."""
    frame = frame.copy()
    for column in frame.select_dtypes(include=["object", "string"]).columns:
        frame[column] = frame[column].map(
            lambda value: "'" + value
            if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@"))
            else value
        )
    return frame.to_csv(index=False).encode("utf-8-sig")
