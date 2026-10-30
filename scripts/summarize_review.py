"""Joint la revue assistée aux textes et produit une synthèse vérifiable."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from sentiment.data import ROOT


def main():
    source = pd.read_json(ROOT / "reports/review_sample.json")
    annotations = pd.read_csv(ROOT / "reports/assisted_annotations.csv")
    assert len(annotations) == 80 and annotations.review_id.is_unique
    assert set(annotations.review_id) == set(source.review_id)
    expected = source.set_index("review_id").text_key.sort_index()
    actual = annotations.set_index("review_id").text_key.sort_index()
    if not expected.equals(actual):
        raise ValueError("Les hashes des textes annotés ne correspondent plus à l'échantillon. Ne pas réattribuer automatiquement les annotations à d'autres avis.")
    result = source.merge(annotations, on=["review_id", "text_key"], validate="one_to_one")
    result["model_error_against_rating"] = result.label.ne(result.prediction)
    result["reviewer"] = "Codex — revue assistée par IA, non indépendante et non aveugle"
    result.to_csv(ROOT / "reports/assisted_review.csv", index=False, encoding="utf-8-sig")
    summary = dict(rows=len(result), rating_model_errors=int(result.model_error_against_rating.sum()),
                   assisted_sentiments=result.assisted_sentiment.value_counts().to_dict(),
                   primary_patterns=result.primary_pattern.value_counts().to_dict(),
                   possible_rating_mismatches=result.loc[result.possible_rating_mismatch, "review_id"].tolist(),
                   outside_food_examples=result.loc[result.outside_food, "review_id"].tolist(),
                   error_patterns=result.loc[result.model_error_against_rating, "primary_pattern"].value_counts().to_dict(),
                   limitations=["Annotations assistées par IA après lecture des textes, non validées par un humain.",
                                "Notes et prédictions visibles pendant la revue : protocole non aveugle.",
                                "20 avis par cellule de confusion : surreprésentation intentionnelle des erreurs.",
                                "Catégorie principale subjective et exclusive ; plusieurs phénomènes peuvent coexister.",
                                "Les comptes décrivent ces 80 avis et ne sont pas des estimations du corpus.",
                                "Le test a maintenant été inspecté ; il ne doit plus guider des ajustements présentés comme évalués indépendamment."])
    (ROOT / "reports/assisted_review_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
