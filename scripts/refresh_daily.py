"""Actualise les agrégats publics sans publier les textes Reddit ni réentraîner sur Reddit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import pandas as pd

from sentiment.data import ROOT, SAMPLE_PATH, MODEL_PATH, AUDIT_PATH
from sentiment.reddit import REDDIT_SNAPSHOT, run_collection
from sentiment.storage import write_json
from sentiment.topics import TOPIC_MODEL, TOPIC_REPORT, TOPIC_ASSIGNMENTS, corpus_fingerprint, assign_topics


def public_reddit_summary(status, frame=None, sentiment_model=None, topic_bundle=None):
    # Liste blanche : aucun texte, titre, auteur, identifiant ou hash individuel dans le site public.
    public = {key: status[key] for key in ["status", "attempted_at", "collected_at", "documents", "source_period", "subreddits", "lookback_days", "error_type"] if key in status}
    public["prediction_scope"] = "Modèles Amazon appliqués sans adaptation à des discussions Reddit ; qualité non évaluée sur Reddit."
    if status["status"] != "ok" or frame is None:
        return public
    public["predicted_positive"] = None
    public["predicted_negative"] = None
    public["topics"] = []
    if frame.empty:
        public.update(predicted_positive=0, predicted_negative=0)
        return public
    labels = sentiment_model.predict(frame.text)
    topics, _ = assign_topics(topic_bundle, frame.text)
    public.update(predicted_positive=int((labels == "Positif").sum()),
                  predicted_negative=int((labels == "Négatif").sum()), unassigned_topics=int((topics < 0).sum()))
    public["topics"] = [dict(label=topic_bundle["names"][int(index)], documents=int(count))
                        for index, count in pd.Series(topics[topics >= 0]).value_counts().items()]
    return public


def make_snapshot(audit, topic_report, reddit_summary, now):
    return dict(processed_at=now.isoformat(), overall_status="partial" if reddit_summary["status"] == "error" else "ok",
                schedule=dict(cron="15 7 * * *", timezone="Europe/Paris"),
                amazon=dict(status="historical_static", latest_review_at=audit["period_end"],
                            sample_documents=audit["sample_rows"], topic_documents=topic_report["documents"],
                            corpus_sha256=topic_report["corpus_sha256"]),
                reddit=reddit_summary,
                notes=["La date de traitement n'est pas la date des données : Amazon reste historique jusqu'en 2012.",
                       "Un statut Reddit non configuré ou en erreur ne représente pas une collecte vide ni une collecte réussie.",
                       "Seuls les agrégats sont exportés ; les instantanés et annotations détaillés restent locaux."])


def execute(script):
    subprocess.run([sys.executable, str(ROOT / "scripts" / script)], check=True, cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-reddit", action="store_true")
    parser.add_argument("--require-reddit", action="store_true")
    args = parser.parse_args()
    if args.skip_reddit and args.require_reddit:
        parser.error("--skip-reddit et --require-reddit sont incompatibles")
    if not SAMPLE_PATH.exists() or not AUDIT_PATH.exists():
        execute("prepare_data.py")
    if not MODEL_PATH.exists():
        execute("train.py")
    frame = pd.read_parquet(SAMPLE_PATH)
    rebuild = not all(path.exists() for path in [TOPIC_MODEL, TOPIC_REPORT, TOPIC_ASSIGNMENTS])
    if not rebuild:
        rebuild = json.loads(TOPIC_REPORT.read_text(encoding="utf-8"))["corpus_sha256"] != corpus_fingerprint(frame)
    if rebuild:
        execute("extract_topics.py")
    now = datetime.now(timezone.utc)
    status = dict(status="disabled", attempted_at=now.isoformat()) if args.skip_reddit else run_collection()
    reddit_frame = pd.read_parquet(REDDIT_SNAPSHOT) if status["status"] == "ok" else None
    summary = public_reddit_summary(status, reddit_frame, joblib.load(MODEL_PATH), joblib.load(TOPIC_MODEL))
    topics = json.loads(TOPIC_REPORT.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    # Exporter les résultats avant le témoin de traitement terminé.
    execute("export_portfolio.py")
    write_json(ROOT / "portfolio/dist/topics.json", topics)
    report = make_snapshot(audit, topics, summary, datetime.now(timezone.utc))
    write_json(ROOT / "data/live/daily_refresh.json", report)
    write_json(ROOT / "portfolio/dist/refresh.json", report)
    print(json.dumps(report, ensure_ascii=True, indent=2))
    if args.require_reddit and status["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
