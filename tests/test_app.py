import pytest
import pandas as pd
from streamlit.testing.v1 import AppTest

from sentiment.data import ROOT, SAMPLE_PATH, MODEL_PATH

pytestmark = pytest.mark.skipif(not SAMPLE_PATH.exists() or not MODEL_PATH.exists(), reason="Artefacts réels requis")


def app():
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=40).run()


def test_exploration_and_empty_filters():
    at = app()
    assert not at.exception
    assert at.metric[0].value == f"{len(pd.read_parquet(SAMPLE_PATH)):,}".replace(",", " ")
    at.text_input(key="product").set_value("INEXISTANT").run()
    assert not at.exception
    assert "Aucun avis" in at.info[0].value


def test_predict_and_reject_empty_review():
    at = app()
    at.radio(key="page").set_value("Tester un avis").run()
    at.form_submit_button[0].click().run()
    assert not at.exception
    assert "trois mots" in at.warning[0].value
    at.text_area(key="review_text").set_value("This coffee tastes wonderful and I love it. Fresh and delicious!")
    at.form_submit_button[0].click().run()
    assert not at.exception
    assert at.metric[0].value == "Positif"
    at.text_area(key="review_text").set_value("This tastes horrible. Disgusting stale food, a complete waste of money.")
    at.form_submit_button[0].click().run()
    assert not at.exception
    assert at.metric[0].value == "Négatif"


def test_evaluation_and_sources_pages():
    at = app()
    for page in ["Évaluation", "Analyse approfondie", "Données et méthode"]:
        at.radio(key="page").set_value(page).run()
        assert not at.exception
        assert at.metric
