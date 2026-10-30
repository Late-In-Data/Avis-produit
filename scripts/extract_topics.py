"""Extraire des thèmes NMF sur l'échantillon Amazon existant."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from sentiment.data import SAMPLE_PATH
from sentiment.topics import build_topics, save_topics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topics", type=int, default=8)
    args = parser.parse_args()
    bundle, assignments, report = build_topics(pd.read_parquet(SAMPLE_PATH), args.topics)
    save_topics(bundle, assignments, report)
    print(json.dumps({"documents": report["documents"], "unassigned": report["unassigned"],
                      "converged": report["converged"], "topics": report["topics"]}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
