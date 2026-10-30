"""Application locale — lancer avec python -m streamlit run app.py."""
from __future__ import annotations

import json
import re

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from sentiment.data import (
    ROOT, SAMPLE_PATH, AUDIT_PATH, METRICS_PATH, MODEL_PATH, SOURCE_PAGE,
    clean_text, safe_csv,
)

st.set_page_config(page_title="Avis produit", page_icon=":material/reviews:", layout="wide")
st.title("Avis produit")
st.caption("Comprendre les avis alimentaires et tester leur sentiment global.")


@st.cache_data(max_entries=2)
def load_data(version: float):
    return pd.read_parquet(SAMPLE_PATH)


@st.cache_resource(max_entries=2)
def load_model(version: float):
    # Artefact local produit par scripts/train.py ; aucun modèle importé par l'utilisateur.
    return joblib.load(MODEL_PATH)


@st.cache_data(max_entries=4)
def load_json(path: str, version: float):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def count_chart(frame, x, y, color=None):
    chart = alt.Chart(frame).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X(f"{x}:O", title=None),
        y=alt.Y(f"{y}:Q", title=y, scale=alt.Scale(zero=True)),
        tooltip=list(frame.columns),
    )
    if color:
        chart = chart.encode(color=alt.Color(
            f"{color}:N", title=None,
            scale=alt.Scale(domain=["Négatif", "3 étoiles (hors modèle)", "Positif"],
                            range=["#C2413A", "#9CA3AF", "#0F766E"])))
    return chart.properties(height=280)


with st.sidebar:
    st.header("Explorer")
    page = st.radio("Navigation", ["Vue d’ensemble", "Tester un avis", "Topics et actualisation", "Évaluation", "Analyse approfondie", "Données et méthode"],
                    label_visibility="collapsed", key="page")
    st.divider()
    st.markdown("**Amazon Fine Foods**")
    st.caption("Avis en anglais · archive historique de 1999 à 2012")
    st.link_button("Consulter la source Stanford", SOURCE_PAGE, icon=":material/open_in_new:")

if not all(path.exists() for path in [SAMPLE_PATH, AUDIT_PATH]):
    st.info("Prépare les données pour ouvrir l’application.")
    st.code(".venv\\Scripts\\python.exe scripts\\prepare_data.py\n.venv\\Scripts\\python.exe scripts\\train.py", language="powershell")
    st.stop()

data = load_data(SAMPLE_PATH.stat().st_mtime)
audit = load_json("reports/data_audit.json", AUDIT_PATH.stat().st_mtime)

if page == "Vue d’ensemble":
    st.subheader("Que disent les notes des clients ?")
    st.caption("Sentiment déduit des étoiles, sans prédiction du modèle. L’échantillon porte sur des textes uniques ; il ne reproduit pas le volume de ventes ou tous les avis de chaque produit.")
    with st.container(horizontal=True):
        product = st.text_input("Identifiant produit exact (facultatif)", placeholder="Ex. B001E4KFG0", key="product")
        years = st.slider("Années", int(data.date.dt.year.min()), int(data.date.dt.year.max()),
                          (int(data.date.dt.year.min()), int(data.date.dt.year.max())), key="years")
    filtered = data.loc[data.date.dt.year.between(*years)]
    if product.strip():
        filtered = filtered.loc[filtered.product_id.eq(product.strip())]
    if filtered.empty:
        st.info("Aucun avis pour ces filtres. Essaie un autre identifiant ou une période plus large.")
        st.stop()
    binary = filtered.loc[filtered.score.ne(3)]
    with st.container(horizontal=True):
        st.metric("Avis dans la sélection", f"{len(filtered):,}".replace(",", " "), border=True)
        st.metric("Positifs parmi les avis binaires", f"{binary.score.ge(4).mean():.1%}" if len(binary) else "—", border=True)
        st.metric("Avis à 3 étoiles exclus du modèle", int(filtered.score.eq(3).sum()), border=True)
        st.metric("Produits représentés", filtered.product_id.nunique(), border=True)
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown("**Répartition des notes**")
            ratings = filtered.groupby(["score", "label"]).size().reset_index(name="Avis")
            ratings = ratings.rename(columns={"score": "Étoiles", "label": "Classe"})
            st.altair_chart(count_chart(ratings, "Étoiles", "Avis", "Classe"),
                            alt="Nombre d’avis par note et classe de sentiment")
    with right:
        with st.container(border=True):
            st.markdown("**Volume de l’échantillon par année**")
            annual = filtered.groupby(filtered.date.dt.year).size().reindex(range(years[0], years[1]+1), fill_value=0).rename_axis("Année").reset_index(name="Avis")
            st.altair_chart(count_chart(annual, "Année", "Avis"), alt="Nombre d’avis échantillonnés par année")
    st.subheader("Produits à examiner")
    st.caption("Tri par nombre d’avis négatifs dans la sélection. Identifiants d’origine : les noms de produits ne sont pas fournis. Un petit effectif ne suffit pas à conclure.")
    minimum = st.slider("Minimum d’avis par produit", 1, 50, 5, key="minimum")
    products = filtered.assign(negative=filtered.score.le(2).astype(int)).groupby("product_id").agg(
        Avis=("score", "size"), Négatifs=("negative", "sum"), Note_moyenne=("score", "mean"))
    products = products.loc[products.Avis.ge(minimum)].sort_values(["Négatifs", "Avis"], ascending=False).head(20).reset_index()
    products.columns = ["Produit", "Avis", "Négatifs", "Note moyenne"]
    products["Note moyenne"] = products["Note moyenne"].round(2)
    st.dataframe(products, hide_index=True, alt="Produits comptant le plus d’avis négatifs dans la sélection")
    st.subheader("Lire les avis")
    with st.container(horizontal=True):
        classes = st.multiselect("Classes", sorted(data.label.unique()), default=sorted(data.label.unique()), key="classes")
        search = st.text_input("Rechercher dans le texte", key="search")
    reviews = filtered.loc[filtered.label.isin(classes)]
    if search.strip():
        reviews = reviews.loc[reviews.text.str.contains(search.strip(), case=False, regex=False)]
    reviews = reviews.sort_values("date", ascending=False)
    st.caption(f"{len(reviews):,} avis correspondent. Aperçu limité aux 200 premiers ; export de toute la sélection.".replace(",", " "))
    visible = reviews[["product_id", "score", "label", "date", "summary", "text"]].rename(columns={
        "product_id": "Produit", "score": "Étoiles", "label": "Classe", "date": "Date", "summary": "Titre", "text": "Avis"})
    st.dataframe(visible.head(200), hide_index=True, alt="Avis correspondant aux filtres de lecture")
    if not visible.empty:
        index = st.selectbox("Lire un avis complet", range(min(200, len(visible))),
                             format_func=lambda i: f"{visible.iloc[i]['Produit']} · {visible.iloc[i]['Étoiles']} étoiles · {visible.iloc[i]['Titre'][:80]}")
        st.text(visible.iloc[index]["Avis"])
    st.download_button("Exporter ces avis", safe_csv(visible), "avis_selection.csv", "text/csv", icon=":material/download:")

elif page == "Tester un avis":
    st.subheader("Prédire positif ou négatif")
    st.info("Écris un avis en anglais sur un produit alimentaire. Le modèle prédit la classe associée aux notes 1–2 ou 4–5 étoiles ; il ne reconnaît pas une classe neutre et ne sépare pas les aspects d’un avis mixte.")
    if not MODEL_PATH.exists():
        st.warning("Entraîne le modèle avec scripts/train.py pour activer les prédictions.")
        st.stop()
    with st.form("prediction"):
        text = st.text_area("Avis en anglais", height=150, max_chars=10000,
                            placeholder="The coffee tastes great and arrived fresh.", key="review_text")
        submitted = st.form_submit_button("Analyser l’avis", type="primary", icon=":material/analytics:")
    if submitted:
        cleaned = clean_text(text)
        if len(re.findall(r"[a-zA-Z]+", cleaned)) < 3:
            st.warning("Saisis au moins trois mots pour obtenir une prédiction exploitable.")
        else:
            model = load_model(MODEL_PATH.stat().st_mtime)
            features = model.named_steps["tfidf"].transform([cleaned])
            if features.nnz == 0:
                st.warning("Aucun terme reconnu par le modèle. Essaie un avis alimentaire en anglais.")
            else:
                label = model.predict([cleaned])[0]
                positive_index = list(model.classes_).index("Positif")
                score = float(model.predict_proba([cleaned])[0, positive_index])
                with st.container(horizontal=True):
                    st.metric("Sentiment prédit", label, border=True)
                    st.metric("Score du modèle pour « positif »", f"{score:.1%}", border=True)
                st.caption("Score non calibré : ce pourcentage n’est pas une garantie de fiabilité. Le seuil de décision est 50 % ; les avis mixtes, le sarcasme et les langues non évaluées peuvent tromper le modèle.")
                if .35 <= score <= .65:
                    st.warning("Le score est proche du seuil : relis cet avis avant de retenir la classe prédite.")

elif page == "Topics et actualisation":
    st.subheader("Quels thèmes reviennent dans les avis ?")
    topic_path = ROOT / "reports/topics.json"
    if topic_path.exists():
        topic_report = load_json("reports/topics.json", topic_path.stat().st_mtime)
        st.caption("Thèmes globaux exploratoires extraits automatiquement avec TF-IDF et NMF sur tout l’échantillon Amazon. Les intitulés reprennent les mots dominants ; ils ne sont pas validés par un humain. La prédiction de sentiment reste positif/négatif.")
        with st.container(horizontal=True):
            st.metric("Thèmes extraits", topic_report["n_topics"], border=True)
            st.metric("Avis analysés", topic_report["documents"], border=True)
            st.metric("Sans composante reconnue", topic_report["unassigned"], border=True)
        topic_frame = pd.DataFrame(topic_report["topics"])
        display = topic_frame[["label", "documents", "positive", "negative", "three_stars"]].rename(columns={
            "label": "Thème dominant", "documents": "Avis", "positive": "Notes 4–5", "negative": "Notes 1–2", "three_stars": "Notes 3"})
        st.dataframe(display, hide_index=True, alt="Nombre d’avis et répartition des notes selon le thème dominant")
        chart = alt.Chart(display).mark_bar().encode(y=alt.Y("Thème dominant:N", title=None, sort="-x"),
            x=alt.X("Avis:Q", scale=alt.Scale(zero=True)), tooltip=list(display.columns))
        st.altair_chart(chart, alt="Nombre d’avis par thème dominant extrait automatiquement")
        selected = st.selectbox("Explorer un thème", topic_frame.id.tolist(),
                                format_func=lambda index: topic_frame.loc[topic_frame.id.eq(index), "label"].iloc[0], key="topic")
        selected_topic = next(row for row in topic_report["topics"] if row["id"] == selected)
        st.markdown("**Mots dominants :** " + ", ".join(selected_topic["keywords"]))
        assignments_path = ROOT / "data/processed/topic_assignments.parquet"
        if assignments_path.exists():
            assignments = pd.read_parquet(assignments_path)
            subset = data.merge(assignments.loc[assignments.topic_id.eq(selected)], on="text_key", validate="one_to_one")
            st.caption("Exemples locaux classés par dominance de la composante. Ce score n’est pas une probabilité ni une mesure validée de pertinence.")
            st.dataframe(subset.sort_values("topic_strength", ascending=False)[["score", "text"]].head(10).rename(columns={"score": "Étoiles", "text": "Avis"}),
                         hide_index=True, alt="Exemples d’avis liés au thème choisi")
        st.download_button("Télécharger la synthèse des thèmes", topic_path.read_bytes(), "topics.json", "application/json")
        st.info("Un avis peut évoquer plusieurs thèmes ; les volumes retiennent seulement la composante dominante. Ces thèmes ne constituent pas une analyse de sentiment par aspect.")
    else:
        st.info("Lance scripts/extract_topics.py pour calculer les thèmes sur les avis Amazon.")
    st.subheader("Actualisation et source Reddit")
    refresh_path = ROOT / "data/live/daily_refresh.json"
    if refresh_path.exists():
        refresh = load_json("data/live/daily_refresh.json", refresh_path.stat().st_mtime)
        processed = pd.Timestamp(refresh["processed_at"]).tz_convert("Europe/Paris").strftime("%d/%m/%Y à %H:%M")
        st.markdown(f"**Dernier traitement terminé :** {processed}, heure de Paris · **Planification :** chaque jour à 7 h 15 via GitHub Actions.")
        st.caption(f"Dernier avis Amazon : {refresh['amazon']['latest_review_at'][:10]}. Recalculer l’application ne crée pas de nouveaux avis dans ce corpus historique.")
        reddit = refresh["reddit"]
        status_labels = {"not_configured": "Identifiants absents — aucune collecte effectuée", "approval_required": "Accès Reddit à approuver", "disabled": "Collecte désactivée", "ok": "Collecte terminée", "error": "Échec de collecte — données non actualisées"}
        st.markdown("**Reddit :** " + status_labels.get(reddit["status"], reddit["status"]))
        if reddit["status"] == "ok":
            st.metric("Discussions collectées", reddit["documents"], border=True)
            st.caption(f"Collecte du {reddit['collected_at']}. Les comptes positifs/négatifs sont des prédictions d’un modèle Amazon, sans validation sur Reddit. Les discussions Reddit ne sont pas des avis produits labellisés.")
            st.json({key: reddit[key] for key in ["predicted_positive", "predicted_negative", "topics"] if key in reddit})
        st.link_button("Consulter les exécutions quotidiennes", "https://github.com/Late-In-Data/Avis-produit/actions/workflows/daily-refresh.yml")
    else:
        st.info("Lance scripts/refresh_daily.py pour recalculer les agrégats et vérifier la configuration Reddit.")

elif page == "Évaluation":
    st.subheader("Mesurer avant de conclure")
    if not METRICS_PATH.exists():
        st.info("Lance scripts/train.py pour afficher les résultats mesurés.")
        st.stop()
    metrics = load_json("reports/metrics.json", METRICS_PATH.stat().st_mtime)
    st.caption(f"Modèle retenu : {metrics['selected_model']}. Choix sur validation, puis mesure sur un test séparé. Référence : toujours prédire la classe majoritaire.")
    with st.container(horizontal=True):
        st.metric("Macro-F1 sur test", f"{metrics['test']['macro_f1']:.3f}", border=True)
        st.metric("Accuracy sur test", f"{metrics['test']['accuracy']:.1%}", border=True)
        st.metric("Macro-F1 de la référence", f"{metrics['baseline']['macro_f1']:.3f}", border=True)
        st.metric("Avis du test", metrics["split_counts"]["test"], border=True)
    st.caption("Le macro-F1 donne le même poids aux deux classes. L’accuracy seule peut masquer les erreurs sur les avis négatifs, moins nombreux.")
    comparison = pd.DataFrame([{"Modèle": name, "Macro-F1 validation": round(result["macro_f1"], 4),
                                "Accuracy validation": round(result["accuracy"], 4)} for name, result in metrics["validation"].items()])
    st.dataframe(comparison, hide_index=True, alt="Comparaison des modèles sur validation")
    left, right = st.columns(2)
    with left:
        st.markdown("**Matrice de confusion du test**")
        matrix = pd.DataFrame(metrics["test"]["confusion_matrix"], index=metrics["labels"], columns=metrics["labels"])
        matrix.index.name = "Classe selon la note ↓ / prédiction →"
        st.dataframe(matrix, alt="Matrice de confusion, lignes réelles et colonnes prédites")
    with right:
        st.markdown("**Qualité par classe sur test**")
        report = pd.DataFrame({label: metrics["test"]["report"][label] for label in metrics["labels"]}).T.round(3)
        st.dataframe(report, alt="Précision, rappel, F1 et effectif pour chaque classe")
    st.subheader("Examiner les erreurs")
    test = pd.read_parquet(ROOT / "data/processed/test_predictions.parquet")
    errors = test.loc[test.label.ne(test.prediction)]
    st.caption(f"{len(errors)} erreurs sur {len(test)} avis du test. La note est une référence imparfaite pour juger le texte.")
    st.dataframe(errors[["score", "label", "prediction", "text"]].head(100).rename(columns={
        "score": "Étoiles", "label": "Selon la note", "prediction": "Prédiction", "text": "Avis"}),
        hide_index=True, alt="Premières erreurs du modèle sur le test")

elif page == "Données et méthode":
    st.subheader("Pourquoi une classification binaire ?")
    st.markdown("La source fournit **un texte et une note globale**, sans labels par aspect. La V1 apprend donc à distinguer **1–2 étoiles (négatif)** de **4–5 étoiles (positif)**. Les **3 étoiles restent visibles dans l’exploration**, mais sont exclues de l’apprentissage et du test. Elles ne sont pas automatiquement considérées comme neutres.")
    st.markdown("Une analyse par aspect demanderait un jeu annoté supplémentaire pour mesurer sa qualité. Elle reste une extension possible après cette V1.")
    with st.container(horizontal=True):
        st.metric("Avis lus dans la source", f"{audit['rows_raw']:,}".replace(",", " "), border=True)
        st.metric("Occurrences de textes déjà vus", f"{audit['duplicate_text_rows']:,}".replace(",", " "), border=True)
        st.metric("Textes aux labels binaires contradictoires", audit["conflicting_binary_texts"], border=True)
    st.markdown("**Traçabilité et exclusions**")
    st.markdown(f"Source : [Stanford SNAP]({SOURCE_PAGE}). Corpus historique du {audit['period_start'][:10]} au {audit['period_end'][:10]}. La date de téléchargement ne correspond pas à la date des avis.")
    st.markdown(f"Échantillonnage reproductible : graine {audit['seed']}, réservoir de {audit['sample_requested']:,} textes uniques, puis exclusion de {audit['conflicts_removed_from_sample']} textes dont les notes se contredisent entre les deux classes. Il reste {audit['sample_rows']:,} avis, dont {audit['sample_binary_rows']:,} utilisables pour le modèle.".replace(",", " "))
    st.markdown("Le nettoyage retire les balises HTML et normalise les espaces. Les doublons sont détectés sur le corps du texte sans tenir compte de la casse. Leur première occurrence est conservée : l’attribution à un produit peut donc être affectée. Les paraphrases peuvent subsister. Les identités et noms des auteurs ne sont pas conservés dans l’échantillon.")
    st.markdown("**Protocole d’évaluation**")
    st.markdown("Séparation stratifiée 70 % entraînement / 15 % validation / 15 % test après déduplication. TF-IDF est ajusté sur l’entraînement uniquement. Régression logistique et Naive Bayes sont comparés sur validation ; le meilleur macro-F1 détermine le modèle évalué sur test. Seul le corps du texte entre dans le modèle.")
    st.warning("Les labels sont des approximations déduites des notes. Le test aléatoire mesure des avis du même corpus ; il ne prouve pas la qualité sur de nouveaux produits, des avis actuels, le français ou des aspects précis. Les proportions affichées décrivent l’échantillon dédupliqué.")
    with st.expander("Audit complet et empreinte de la source"):
        st.json(audit)
    st.download_button("Télécharger l’audit des données", AUDIT_PATH.read_bytes(), "data_audit.json", "application/json")

else:
    st.subheader("Aller au-delà du test initial")
    st.caption("Évaluations rétrospectives supplémentaires sur le même échantillon. Elles recoupent les avis de la V1 : ce ne sont pas de nouvelles validations externes indépendantes.")
    robustness_path = ROOT / "reports/robustness.json"
    if robustness_path.exists():
        robustness = load_json("reports/robustness.json", robustness_path.stat().st_mtime)
        initial = load_json("reports/metrics.json", METRICS_PATH.stat().st_mtime)
        rows = [{"Protocole": "Test aléatoire initial", "Avis de test": initial["split_counts"]["test"],
                 "Accuracy": initial["test"]["accuracy"], "Macro-F1": initial["test"]["macro_f1"]}]
        for key, label in [("temporal", "Période postérieure"), ("unseen_products", "Produits inconnus")]:
            result = robustness[key]
            rows.append({"Protocole": label, "Avis de test": result["counts"]["test"],
                         "Accuracy": result["test"]["accuracy"], "Macro-F1": result["test"]["macro_f1"]})
        summary = pd.DataFrame(rows)
        st.dataframe(summary.round(4), hide_index=True, alt="Qualité du modèle lexical selon trois protocoles")
        chart = alt.Chart(summary).mark_bar().encode(
            y=alt.Y("Protocole:N", title=None), x=alt.X("Macro-F1:Q", scale=alt.Scale(domain=[0, 1])),
            tooltip=["Protocole", "Avis de test", "Accuracy", "Macro-F1"])
        st.altair_chart(chart, alt="Macro-F1 du modèle lexical selon le protocole avec une échelle de zéro à un")
        temporal = robustness["temporal"]
        st.markdown(f"**Période postérieure** : entraînement jusqu’au {temporal['periods']['train'][1][:10]}, validation réservée ensuite et test du {temporal['periods']['test'][0][:10]} au {temporal['periods']['test'][1][:10]}. Aucun chevauchement de dates entre ensembles.")
        st.markdown("**Produits inconnus** : séparation par identifiant produit (ASIN), sans ASIN partagé entre les ensembles. Les variantes et familles de produits ne sont pas regroupées. Les hyperparamètres V1 sont figés pour ces deux tests ; le modèle de démonstration n’est pas remplacé.")
    benchmark_path = ROOT / "reports/transformer_benchmark.json"
    if benchmark_path.exists():
        benchmark = load_json("reports/transformer_benchmark.json", benchmark_path.stat().st_mtime)
        st.subheader("Comparaison avec DistilBERT SST-2")
        comparison = pd.DataFrame([
            {"Modèle": "TF-IDF + régression logistique", "Macro-F1": benchmark["logistic"]["metrics"]["macro_f1"],
             "Accuracy": benchmark["logistic"]["metrics"]["accuracy"], "ms par avis (débit)": benchmark["logistic"]["milliseconds_per_review"],
             "Artefact / poids (Mo)": benchmark["logistic"]["artifact_mb"]},
            {"Modèle": "DistilBERT SST-2", "Macro-F1": benchmark["transformer"]["metrics"]["macro_f1"],
             "Accuracy": benchmark["transformer"]["metrics"]["accuracy"], "ms par avis (débit)": benchmark["transformer"]["milliseconds_per_review"],
             "Artefact / poids (Mo)": benchmark["transformer"]["weights_mb"]}])
        st.dataframe(comparison.round(4), hide_index=True, alt="Qualité, débit CPU et poids des artefacts de deux stratégies de classification")
        st.caption(f"Même test de {benchmark['test_rows']} avis. DistilBERT est préentraîné sur SST-2, sans adaptation Amazon ; {benchmark['truncated_reviews']} textes sont tronqués aux {benchmark['max_length']} premiers tokens. Débit mesuré sur CPU après échauffement, tokenisation comprise ; ce n’est pas une latence de requête isolée. Le transformer est calculé par lots, le modèle lexical sur le test complet.")
        st.caption(f"Mémoire observée du processus de benchmark : {benchmark['transformer']['observed_process_rss_mb']:.0f} Mo. Inclut Torch, les données et le modèle ; ne mesure pas la mémoire isolée du transformer.")
        st.link_button("Lire la fiche du modèle comparé", f"https://huggingface.co/{benchmark['model_id']}")
        interval_path = ROOT / "reports/model_comparison.json"
        if interval_path.exists():
            interval = load_json("reports/model_comparison.json", interval_path.stat().st_mtime)
            low, high = interval["percentile_interval_95"]
            st.caption(f"Écart de macro-F1 (transformer − lexical) : {interval['difference_transformer_minus_lexical']:+.4f}, intervalle bootstrap apparié à 95 % [{low:+.4f} ; {high:+.4f}]. Rééchantillonnage par avis ; les dépendances entre auteurs et produits ne sont pas corrigées.")
    review_path = ROOT / "reports/assisted_review.csv"
    if review_path.exists():
        st.subheader("Revue qualitative de 80 avis")
        st.warning("Revue assistée par IA, non validée par un humain et non aveugle aux notes/prédictions. 20 avis par cellule de confusion, dont 40 erreurs : les proportions ne représentent pas le corpus.")
        reviewed = pd.read_csv(review_path)
        review_summary = load_json("reports/assisted_review_summary.json", (ROOT / "reports/assisted_review_summary.json").stat().st_mtime)
        with st.container(horizontal=True):
            st.metric("Avis relus avec assistance IA", len(reviewed), border=True)
            st.metric("Désaccords note / texte possibles", len(review_summary["possible_rating_mismatches"]), border=True)
            st.metric("Exemples hors alimentation", len(review_summary["outside_food_examples"]), border=True)
        st.caption("Les deux désaccords possibles et les exemples hors alimentation sont des observations sur ces 80 avis, sans extrapolation au corpus entier. Ils justifient une annotation humaine et un contrôle du périmètre avant une nouvelle version.")
        selected = st.selectbox("Avis de la revue", reviewed.review_id.tolist(), key="review_case")
        row = reviewed.loc[reviewed.review_id.eq(selected)].iloc[0]
        st.text(row.text)
        st.markdown(f"**Selon la note :** {row.label} · **Modèle :** {row.prediction} · **Lecture assistée :** {row.assisted_sentiment}")
        st.text(row.reason)
        st.download_button("Exporter la revue assistée", safe_csv(reviewed), "revue_assistee.csv", "text/csv")
        st.download_button("Exporter le formulaire de revue humaine", (ROOT / "reports/human_review_template.csv").read_bytes(), "revue_humaine_a_completer.csv", "text/csv")
        st.info("La revue fait apparaître des avis mixtes et des limites de labels ; elle ne suffit pas à valider une analyse par aspect. La prédiction reste binaire. Le test relu ne doit plus servir à régler le modèle tout en étant présenté comme indépendant.")
