"""Collecte PRAW bornée, en lecture seule, avec instantanés locaux remplaçables."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
import re
import tomllib

import pandas as pd

from sentiment.data import ROOT, clean_text, text_key
from sentiment.storage import write_json, write_parquet

REDDIT_SNAPSHOT = ROOT / "data/live/reddit.parquet"
REDDIT_STATUS = ROOT / "data/live/reddit_status.json"
COLUMNS = ["id", "subreddit", "created_at", "collected_at", "title", "text", "url", "text_key"]


def credentials(environ=None, secrets_path=None):
    environ = os.environ if environ is None else environ
    secrets_path = ROOT / ".streamlit/secrets.toml" if secrets_path is None else secrets_path
    local = {}
    if secrets_path.exists():
        local = tomllib.loads(secrets_path.read_text(encoding="utf-8")).get("reddit", {})
    config = {key: str(environ.get("REDDIT_" + key.upper(), local.get(key, ""))).strip()
              for key in ["client_id", "client_secret", "user_agent"]}
    approved = str(environ.get("REDDIT_ACCESS_APPROVED", local.get("access_approved", False))).lower() == "true"
    missing = ["REDDIT_"+key.upper() for key, value in config.items() if not value]
    state = "not_configured" if missing else "approval_required" if not approved else "ready"
    return config, dict(status=state, missing_variables=missing, access_approved=approved)


def load_config(path=None):
    path = ROOT / "config/reddit.toml" if path is None else path
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    names = config.get("subreddits", [])
    if not names or len(names) > 5 or any(not re.fullmatch(r"[A-Za-z0-9_]{2,21}", name) for name in names):
        raise ValueError("Configurer entre 1 et 5 noms de subreddits valides.")
    if not 1 <= config.get("limit_per_subreddit", 0) <= 200:
        raise ValueError("La limite par subreddit doit être comprise entre 1 et 200.")
    if not 1 <= config.get("lookback_days", 0) <= 30:
        raise ValueError("Choisir une fenêtre de 1 à 30 jours.")
    if not config.get("query", "").strip():
        raise ValueError("La recherche ne peut pas être vide.")
    return config


def collect_snapshot(client, config, now=None):
    """Fonction injectable pour tests hors réseau. Aucune identité d'auteur conservée."""
    now = datetime.now(timezone.utc) if now is None else now
    cutoff = now - timedelta(days=config["lookback_days"])
    time_filter = "week" if config["lookback_days"] <= 7 else "month"
    records = []
    for name in config["subreddits"]:
        submissions = client.subreddit(name).search(config["query"], sort="new", time_filter=time_filter,
                                                   limit=config["limit_per_subreddit"])
        for submission in submissions:
            body = str(getattr(submission, "selftext", "") or "").strip()
            if body in ("", "[removed]", "[deleted]") or getattr(submission, "over_18", False) or getattr(submission, "stickied", False):
                continue
            if getattr(submission, "author", None) is None:
                continue
            date = datetime.fromtimestamp(float(submission.created_utc), tz=timezone.utc)
            if not cutoff <= date <= now + timedelta(minutes=5):
                continue
            title = clean_text(getattr(submission, "title", ""))
            # Ne pas assimiler les votes Reddit à une note client ou à un label de sentiment.
            text = clean_text(title + ". " + body)
            if len(re.findall(r"[A-Za-z]+", text)) < 5:
                continue
            records.append(dict(id=str(submission.id), subreddit=name, created_at=date.isoformat(),
                                collected_at=now.isoformat(), title=title, text=text,
                                url="https://www.reddit.com"+str(submission.permalink), text_key=text_key(text)))
    frame = pd.DataFrame(records, columns=COLUMNS)
    if not frame.empty:
        frame = frame.drop_duplicates("id").drop_duplicates("text_key").reset_index(drop=True)
    return frame


def run_collection(environ=None, secrets_path=None, client_factory=None, config=None, now=None):
    now = datetime.now(timezone.utc) if now is None else now
    auth, status = credentials(environ, secrets_path)
    status["attempted_at"] = now.isoformat()
    status["source"] = "Reddit Data API via PRAW — discussions, sans labels de vérité"
    if status["status"] != "ready":
        write_json(REDDIT_STATUS, status)
        return status
    config = load_config() if config is None else config
    client = None
    try:
        if client_factory is None:
            import praw
            client_factory = praw.Reddit
        client = client_factory(**auth, check_for_async=False, check_for_updates=False, timeout=15)
        client.read_only = True
        frame = collect_snapshot(client, config, now)
        # Remplacement complet, sans historique de posts ou réutilisation d'anciennes lignes absentes.
        write_parquet(REDDIT_SNAPSHOT, frame)
        status.update(status="ok", collected_at=now.isoformat(), documents=len(frame),
                      source_period=[frame.created_at.min(), frame.created_at.max()] if len(frame) else None,
                      subreddits=config["subreddits"], lookback_days=config["lookback_days"])
    except Exception as error:
        # Aucune chaîne d'exception ni réponse HTTP : elles pourraient contenir des secrets.
        status.update(status="error", error_type=type(error).__name__, documents=None)
    finally:
        # PRAW 8 expose un contexte et close selon la version ; fermer si disponible.
        if client is not None:
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
    write_json(REDDIT_STATUS, status)
    return status
