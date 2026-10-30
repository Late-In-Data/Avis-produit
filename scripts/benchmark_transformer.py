"""Comparer un DistilBERT SST-2 figé à la V1 sur le même test, sur CPU."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sentiment.data import ROOT, MODEL_PATH

# Les téléchargements et caches restent dans le projet, sans envoi des avis à une API.
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import joblib
import pandas as pd
import psutil
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scripts.train import evaluate

MODEL_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
MODEL_REVISION = "714eb0fa89d2f80546fda750413ed43d93601a13"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    test = pd.read_parquet(ROOT / "data/processed/test_predictions.parquet")
    revision = MODEL_REVISION
    print(f"Téléchargement {MODEL_ID}@{revision}", flush=True)
    started = time.perf_counter()
    snapshot = snapshot_download(MODEL_ID, revision=revision,
                                 local_dir=ROOT / ".cache/transformer" / revision,
                                 allow_patterns=["config.json", "model.safetensors", "tokenizer_config.json", "tokenizer.json", "vocab.txt", "special_tokens_map.json"])
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(snapshot, local_files_only=True,
                                                               trust_remote_code=False, use_safetensors=True).eval()
    loading_seconds = time.perf_counter()-started
    labels = {int(k): ("Positif" if v.upper() == "POSITIVE" else "Négatif") for k, v in model.config.id2label.items()}
    assert set(model.config.id2label.values()) == {"NEGATIVE", "POSITIVE"}
    texts = test.text.tolist()
    lengths = [len(ids) for ids in tokenizer(texts, truncation=False, add_special_tokens=True)["input_ids"]]
    warmup = tokenizer(texts[:args.batch_size], return_tensors="pt", padding=True,
                       truncation=True, max_length=args.max_length)
    with torch.inference_mode():
        model(**warmup)
    predictions, probabilities = [], []
    started = time.perf_counter()
    peak_rss = psutil.Process().memory_info().rss
    positive_id = next(index for index, label in labels.items() if label == "Positif")
    with torch.inference_mode():
        for index in range(0, len(texts), args.batch_size):
            inputs = tokenizer(texts[index:index+args.batch_size], return_tensors="pt", padding=True,
                               truncation=True, max_length=args.max_length)
            logits = model(**inputs).logits
            predictions.extend(labels[int(i)] for i in logits.argmax(dim=-1))
            probabilities.extend(logits.softmax(dim=-1)[:, positive_id].tolist())
            peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
            if index % (args.batch_size*25) == 0:
                print(f"Inférence {min(index+args.batch_size, len(test))}/{len(test)}", flush=True)
    elapsed = time.perf_counter()-started
    logistic = joblib.load(MODEL_PATH)
    logistic.predict(texts[:args.batch_size])
    started = time.perf_counter()
    logistic_predictions = logistic.predict(texts)
    logistic_elapsed = time.perf_counter()-started
    assert list(logistic_predictions) == test.prediction.tolist()
    result = dict(model_id=MODEL_ID, revision=revision, device="cpu", threads=args.threads,
                  torch_version=torch.__version__, processor=platform.processor(),
                  protocol="Modèle SST-2 prédéfini, sans adaptation Amazon ni réglage sur le test. Même test de 5551 avis que la V1. Pas de sélection d'un nouveau champion sur ces résultats.",
                  test_rows=len(test), batch_size=args.batch_size, max_length=args.max_length,
                  truncated_reviews=sum(length>args.max_length for length in lengths),
                  transformer=dict(metrics=evaluate(test.label, predictions), inference_seconds=elapsed,
                                   milliseconds_per_review=1000*elapsed/len(test),
                                   load_and_download_seconds=loading_seconds,
                                   observed_process_rss_mb=peak_rss/1024**2,
                                   weights_mb=sum(p.stat().st_size for p in Path(snapshot).glob("*.safetensors"))/1024**2),
                  logistic=dict(metrics=evaluate(test.label, logistic_predictions), inference_seconds=logistic_elapsed,
                                milliseconds_per_review=1000*logistic_elapsed/len(test),
                                artifact_mb=MODEL_PATH.stat().st_size/1024**2),
                  limitations=["DistilBERT est adapté à SST-2, pas entraîné sur les avis Amazon : comparaison de deux stratégies, pas de familles optimisées également.",
                               "Troncature aux 256 premiers tokens pour le transformer ; le modèle lexical utilise le texte complet.",
                               "Débit en lots, tokenisation incluse, après échauffement ; ne représente pas la latence d'une requête isolée.",
                               "RSS observée du processus complet (Torch, données et modèle), pas la mémoire isolée du seul transformer.",
                               "Le test déjà inspecté reste une référence de comparaison ; toute optimisation ultérieure demandera un nouveau test."])
    (ROOT / "reports/transformer_benchmark.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    test.assign(transformer_prediction=predictions, transformer_positive_score=probabilities).to_parquet(ROOT / "data/processed/transformer_predictions.parquet", index=False)
    print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
