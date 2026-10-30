"""Collecte Reddit réelle uniquement avec identifiants et accès approuvé."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sentiment.reddit import run_collection

if __name__ == "__main__":
    status = run_collection()
    print(json.dumps(status, ensure_ascii=True, indent=2))
    raise SystemExit(0 if status["status"] == "ok" else 2)
