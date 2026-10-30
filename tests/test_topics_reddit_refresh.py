from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
import json

import pandas as pd
import pytest

from sentiment import reddit
from sentiment.topics import build_topics, assign_topics
from scripts.refresh_daily import public_reddit_summary, make_snapshot


def test_praw_client_is_read_only_without_contacting_reddit():
    import praw
    client = praw.Reddit(client_id="fake-test-id", client_secret="fake-test-secret",
                         user_agent="windows:avis-produit-tests:v1.0 (by /u/test)",
                         check_for_async=False, check_for_updates=False, timeout=15)
    assert client.read_only is True


def test_topics_preserve_counts_and_handle_unknown_terms():
    texts = ["coffee roast beans espresso cup strong smooth"]*15 + ["green tea leaf iced tea bags herbal"]*15
    frame = pd.DataFrame({"text": texts, "text_key": [str(i) for i in range(30)], "score": [1, 3, 5]*10})
    model, assignments, report = build_topics(frame, n_topics=2)
    assert report["documents"] == sum(row["documents"] for row in report["topics"]) + report["unassigned"]
    assert sum(row["positive"]+row["negative"]+row["three_stars"] for row in report["topics"]) == 30
    assert len(assignments) == 30
    labels, strengths = assign_topics(model, [texts[0], texts[-1], "zzzzunrecognized"])
    assert labels[0] != labels[1]
    assert labels[2] == -1 and strengths[2] == 0


def post(identifier, now, body="I enjoyed this fresh coffee and its rich roast", **kwargs):
    return SimpleNamespace(id=identifier, title="Coffee review", selftext=body,
                           created_utc=now.timestamp(), author=object(), over_18=False, stickied=False,
                           permalink=f"/r/Coffee/comments/{identifier}/review/", **kwargs)


def test_collect_snapshot_filters_deleted_old_and_duplicate_posts():
    now = datetime(2026, 10, 30, tzinfo=timezone.utc)
    rows = [post("a", now), post("a", now), post("b", now, "[removed]"), post("c", now-timedelta(days=20))]
    calls = []
    def search(query, **kwargs):
        calls.append(kwargs)
        return iter(rows)
    client = SimpleNamespace(subreddit=lambda name: SimpleNamespace(search=search))
    frame = reddit.collect_snapshot(client, dict(subreddits=["Coffee"], query="review", limit_per_subreddit=100, lookback_days=7), now)
    assert frame.id.tolist() == ["a"]
    assert "author" not in frame and "score" not in frame
    assert calls[0] == {"sort": "new", "time_filter": "week", "limit": 100}


def test_unconfigured_collection_never_contacts_api(tmp_path, monkeypatch):
    monkeypatch.setattr(reddit, "REDDIT_STATUS", tmp_path / "status.json")
    def forbidden(**kwargs):
        pytest.fail("Une API ne doit pas être appelée sans identifiants")
    status = reddit.run_collection(environ={}, secrets_path=tmp_path/"missing.toml", client_factory=forbidden)
    assert status["status"] == "not_configured"
    assert "collected_at" not in status


def test_access_approval_required_even_with_credentials(tmp_path):
    auth, status = reddit.credentials(dict(REDDIT_CLIENT_ID="fake-id", REDDIT_CLIENT_SECRET="fake-secret", REDDIT_USER_AGENT="test-agent"), tmp_path/"missing")
    assert status["status"] == "approval_required"


def test_collection_error_preserves_snapshot_and_does_not_leak_secrets(tmp_path, monkeypatch):
    snapshot = tmp_path/"reddit.parquet"
    snapshot.write_bytes(b"previous snapshot")
    monkeypatch.setattr(reddit, "REDDIT_SNAPSHOT", snapshot)
    monkeypatch.setattr(reddit, "REDDIT_STATUS", tmp_path/"status.json")
    def broken(**kwargs):
        raise RuntimeError("NEVER_EXPORT_THIS_SECRET")
    status = reddit.run_collection(environ=dict(REDDIT_CLIENT_ID="fake-id", REDDIT_CLIENT_SECRET="NEVER_EXPORT_THIS_SECRET", REDDIT_USER_AGENT="test-agent", REDDIT_ACCESS_APPROVED="true"),
                                   secrets_path=tmp_path/"missing", client_factory=broken)
    assert status["status"] == "error"
    assert snapshot.read_bytes() == b"previous snapshot"
    assert "NEVER_EXPORT" not in json.dumps(status)
    assert "collected_at" not in status


def test_public_snapshot_distinguishes_processing_time_from_data_date():
    summary = public_reddit_summary({"status": "not_configured", "title": "PRIVATE", "text": "PRIVATE", "id": "PRIVATE", "token": "PRIVATE"})
    assert "PRIVATE" not in json.dumps(summary)
    report = make_snapshot({"period_end": "2012-10-26", "sample_rows": 40},
                           {"documents": 40, "corpus_sha256": "test"}, summary,
                           datetime(2026, 10, 30, tzinfo=timezone.utc))
    assert report["amazon"]["status"] == "historical_static"
    assert report["amazon"]["latest_review_at"] == "2012-10-26"
    assert "documents" not in report["reddit"]
    assert report["processed_at"].startswith("2026-10-30")
