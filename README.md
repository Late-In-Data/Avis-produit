# Avis produit — sentiment global des avis clients

Une première application de portfolio qui explore de vrais avis alimentaires Amazon et prédit **positif / négatif** à partir du texte anglais. Interface en français, données et prédictions en local, aucun compte ni clé API.

![Présentation du projet](docs/screenshots/portfolio.jpg)

![Application Streamlit](docs/screenshots/streamlit.jpg)

Le projet comprend maintenant des stress tests temporels et par produit inconnu, une revue qualitative assistée de 80 avis, une comparaison avec un transformer préentraîné et une démo du modèle dans le navigateur. Le modèle interactif reste binaire.

**[Essayer la démonstration web](https://late-in-data.github.io/Avis-produit/)** · [Code et rapports](https://github.com/Late-In-Data/Avis-produit) · [Tests automatisés](https://github.com/Late-In-Data/Avis-produit/actions)

## Ouvrir la version installée

Depuis le dossier `Avis-produit`, dans PowerShell :

```powershell
.\lancer.ps1
```

Ou directement, si l'exécution des scripts PowerShell est désactivée :

```powershell
.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

Ouvrir ensuite http://localhost:8501. Le terminal doit rester ouvert ; `Ctrl+C` arrête l'application.

## Ce que fait la V1

- **Vue d'ensemble** : distribution des notes, volume historique, produits à examiner, recherche, lecture des avis et export CSV de la sélection.
- **Tester un avis** : classification binaire d'un texte anglais par un modèle entraîné sur les données réelles.
- **Évaluation** : comparaison sur validation, résultats sur un test indépendant, référence majoritaire, métriques par classe, matrice de confusion et exemples d'erreurs.
- **Analyse approfondie** : stress tests complémentaires, revue assistée, formulaire de revue humaine et benchmark transformer lorsqu'il a été calculé.
- **Données et méthode** : source, règles de labels, exclusions, audit, limites et empreinte SHA-256.

## Pourquoi conserver le sentiment global ?

La source proposée par le projet initial est [Amazon Fine Food Reviews sur Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews). On utilise ici sa [source académique Stanford SNAP](https://snap.stanford.edu/data/web-FineFoods.html), accessible sans compte Kaggle.

Elle fournit un corps d'avis, un résumé, un identifiant produit, une date et une note globale. Elle ne fournit pas de sentiment annoté par aspect. La V1 adopte donc une **convention explicite**, pas une vérité annotée manuellement :

| Note | Usage |
| --- | --- |
| 1–2 étoiles | Négatif |
| 3 étoiles | Conservé dans l'exploration, exclu de l'apprentissage et de l'évaluation |
| 4–5 étoiles | Positif |

Trois étoiles ne signifie pas nécessairement « neutre ». Un avis mixte reste difficile à représenter par un seul label. Une extension par aspect nécessitera des exemples annotés pour pouvoir être évaluée ; elle n'est pas présentée comme une capacité de cette version.

## Installer et reproduire

Python 3.12 recommandé. Sur cette machine, `.venv` et les dépendances sont déjà installés. Pour une installation neuve avec [uv](https://docs.astral.sh/uv/) :

```powershell
uv venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r requirements-dev.txt
.venv\Scripts\python.exe scripts\prepare_data.py
.venv\Scripts\python.exe scripts\train.py
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

Pour reproduire les étapes complémentaires :

```powershell
.venv\Scripts\python.exe scripts\evaluate_robustness.py
.venv\Scripts\python.exe scripts\prepare_review.py
if (Test-Path reports\assisted_annotations.csv) { .venv\Scripts\python.exe scripts\summarize_review.py }
uv pip install --python .venv\Scripts\python.exe -r requirements-evaluation.txt
.venv\Scripts\python.exe scripts\benchmark_transformer.py
.venv\Scripts\python.exe scripts\compare_models.py
.venv\Scripts\python.exe scripts\export_portfolio.py
.venv\Scripts\python.exe scripts\check_export.py
```

Le benchmark télécharge ~255 Mo de poids DistilBERT et s'exécute sur CPU. Les textes ne sont pas envoyés à une API d'inférence. La révision exacte du modèle est figée dans le script et conservée dans le rapport. La démo web se lance séparément avec `python -m http.server 8502 --directory portfolio/dist` ; Node.js est requis uniquement pour le contrôle de parité Python/JavaScript. `requirements-evaluation-lock.txt` conserve les versions de l'environnement expérimental.

La préparation télécharge l'archive (~122 Mo), parcourt les **568 454 avis**, audite le corpus complet et tire un réservoir uniforme de 40 000 textes uniques avec graine 42. L'encodage historique est Latin-1. Aucune donnée synthétique n'est utilisée. Le téléchargement exige Internet ; après préparation et entraînement, l'application fonctionne hors ligne.

Pour reproduire exactement les versions installées lors de la réalisation, utiliser `requirements-lock.txt` à la place de `requirements-dev.txt`.

## Qualité des données et protocole

Les scripts conservent les calculs, les règles et les sources inspectables :

1. Nettoyer les balises HTML, décoder les entités et normaliser les espaces.
2. Détecter les textes identiques sans tenir compte de la casse dans le corpus entier. Conserver la première occurrence et tirer un échantillon uniforme de ces textes uniques.
3. Exclure les textes associés à la fois à des notes 1–2 et 4–5 dans la source entière. Les notes à 3 étoiles restent hors modèle.
4. Séparer les avis binaires en entraînement (70 %), validation (15 %) et test (15 %), par tirage stratifié reproductible. Aucun hash de texte ne figure dans deux ensembles.
5. Ajuster TF-IDF sur l'entraînement uniquement, avec unigrammes/bigrammes et 60 000 variables au maximum. Aucun résumé, note, produit, auteur ou date n'entre dans les variables explicatives.
6. Comparer régression logistique et Naive Bayes sur le macro-F1 de validation. Évaluer le modèle retenu une seule fois sur test, en regard d'une référence qui prédit toujours la classe majoritaire.
7. Sauvegarder exactement le modèle évalué, les métriques, les erreurs et la liste des affectations aux ensembles.

Le modèle n'est pas réentraîné sur le test. Le score affiché lors d'une prédiction n'est pas calibré. Le macro-F1 complète l'accuracy car les classes sont déséquilibrées.

### Résultats de la première exécution

Audit intégral : **568 454 avis**, dont **175 334 occurrences de textes déjà vus** et **7 textes portant des labels binaires contradictoires**. Répartition brute : 82 037 avis à 1–2 étoiles, 42 640 à 3 étoiles et 443 777 à 4–5 étoiles. Aucun champ requis manquant ni avis invalide selon les contrôles implémentés.

Après échantillonnage et retrait d'un texte contradictoire présent dans le réservoir : **39 999 avis**, dont **37 003 binaires** et **2 996 à 3 étoiles**. Régression logistique retenue sur validation (macro-F1 : 0,860 ; Naive Bayes : 0,615).

Sur **5 551 avis de test** :

| Mesure | Modèle retenu | Toujours positif |
| --- | ---: | ---: |
| Accuracy | 92,65 % | 84,44 % |
| Macro-F1 | 0,869 | 0,458 |
| Rappel négatif | 84,61 % | 0 % |
| Précision négative | 72,66 % | 0 % |

Ces résultats évaluent la reproduction des classes déduites des étoiles dans ce corpus. Ils ne mesurent pas un sentiment humainement annoté. Matrice de confusion : 731 négatifs et 4 412 positifs correctement prédits ; 133 négatifs classés positifs et 275 positifs classés négatifs.

### Tests de robustesse supplémentaires

| Protocole | Avis du test | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| Test aléatoire initial | 5 551 | 92,65 % | 0,869 |
| Période postérieure à l'entraînement | 5 571 | 92,43 % | 0,874 |
| Produits inconnus de l'entraînement | 5 441 | 92,24 % | 0,860 |

Le test temporel suit un entraînement jusqu'au 8 février 2012, une validation réservée du 9 février au 26 juin 2012 et un test du 27 juin au 26 octobre 2012. Aucune journée n'est partagée entre ces ensembles. Le test de produits inconnus sépare les ensembles par ASIN, sans ASIN partagé. Les hyperparamètres de la V1 restent figés ; les modèles expérimentaux ne remplacent pas le pipeline interactif sauvegardé.

**Ces expériences réutilisent le même échantillon et recoupent les populations de la V1** : ce sont des stress tests rétrospectifs, pas de nouvelles validations externes. Les produits du test temporel peuvent avoir été vus à l'entraînement. Les familles et variantes de produits ne sont pas regroupées dans le test par ASIN. Les différences de métriques entre protocoles ne mesurent pas un effet causal.

### Revue qualitative assistée

80 avis ont été lus intégralement avec assistance IA, à raison de 20 par cellule de confusion : 40 erreurs et 40 prédictions conformes à la note. Les annotations détaillées sont conservées localement dans `reports/assisted_annotations.csv`, consultables dans Streamlit. La synthèse et la [méthode](docs/revue-qualitative.md) sont publiques.

La lecture assistée repère notamment des négations, des comparaisons, des aspects opposés, deux contradictions possibles entre une note élevée et un texte négatif, ainsi que dix exemples hors alimentation. Les proportions de ce sous-ensemble enrichi en erreurs ne s'extrapolent pas au corpus. **Ce n'est pas une annotation humaine indépendante.** Un formulaire sans note ni prédiction se génère pour une future revue humaine ; il n'a pas été rempli au nom d'un humain.

Le test initial a maintenant été inspecté. Toute optimisation motivée par ces observations doit être évaluée sur un nouveau jeu réservé. La classification reste positif/négatif ; aucune capacité par aspect n'est revendiquée.

### Comparaison avec un transformer préentraîné

DistilBERT SST-2 est comparé sans adaptation Amazon sur les **mêmes 5 551 avis** du test initial. Révision figée : `714eb0fa89d2f80546fda750413ed43d93601a13`.

| Mesure | TF-IDF + régression logistique | DistilBERT SST-2 |
| --- | ---: | ---: |
| Accuracy | 92,65 % | 84,24 % |
| Macro-F1 | 0,869 | 0,772 |
| Rappel négatif | 84,61 % | 91,90 % |
| Précision négative | 72,66 % | 49,66 % |
| Débit CPU, ms par avis | 0,29 | 193,53 |
| Artefact lexical / poids transformer | 2,71 Mo | 255,43 Mo |

DistilBERT repère plus d'avis négatifs (794 contre 731), mais classe à tort 805 avis positifs comme négatifs, contre 275 pour la régression. Le macro-F1 baisse de **0,0971** ; intervalle bootstrap apparié à 95 % sur cette différence : **[−0,1114 ; −0,0821]**, avec 2 000 réplications par avis et graine 2026. Ce bootstrap suppose des avis indépendants et ne corrige pas les dépendances entre auteurs/produits.

Mesure sur CPU AMD64, 4 threads PyTorch, après échauffement et tokenisation comprise : 1 074,29 secondes pour DistilBERT par lots de 8, contre 1,62 seconde pour le pipeline lexical sur le test complet. **Il s'agit d'un débit moyen, pas d'une latence de requête isolée.** 313 textes sont tronqués aux 256 premiers tokens pour DistilBERT ; le modèle lexical utilise tout le texte. La RSS observée du processus de benchmark (~427 Mo) inclut Torch, les données et le modèle et ne constitue pas la mémoire isolée du transformer.

**Décision : conserver le pipeline lexical en démonstration.** Cette expérience compare un modèle adapté au corpus et un modèle préentraîné SST-2 sans adaptation ; elle ne prouve pas qu'un transformer adapté à Amazon serait inférieur. Aucun nouveau champion n'est sélectionné par réglage sur ce test déjà inspecté. [Fiche officielle DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english).

### Démonstration web et inférence locale

`portfolio/dist/` contient une présentation autonome du projet et une démonstration utilisant le même vocabulaire, les mêmes idf et les mêmes coefficients que le modèle Python. Les textes saisis restent dans le navigateur. Les fichiers `results.json` et `model.json` sont régénérés avec `scripts/export_portfolio.py`.

Le contrôle de parité sur les 5 551 textes du test et quatre cas de bord constate **zéro différence de classe**, avec un écart maximal de probabilité inférieur à 10⁻¹⁰. Ce contrôle vérifie l'implémentation exportée sur ces textes ; il ne constitue pas une nouvelle mesure de qualité du modèle.

### Exécution conteneurisée

```powershell
docker build -t avis-produit .
docker run --rm -p 8501:8501 avis-produit
```

La construction télécharge les données originales et entraîne le modèle ; elle demande Internet. Le Dockerfile est fourni pour un hébergement compatible Python/Streamlit. La démonstration Sites est une application navigateur distincte : elle ne lance pas le serveur Streamlit. Aucun Docker n'étant installé dans l'environnement de réalisation, l'image n'a pas été construite ici.

### Validation continue

GitHub Actions exécute les tests sans télécharger le corpus à chaque push. Les tests d'intégration qui exigent des artefacts locaux sont alors ignorés explicitement. Un lancement manuel avec `full_pipeline=true` prépare les vraies données, entraîne la V1, exécute les stress tests et relance les tests. Le benchmark transformer reste une commande expérimentale séparée.

### Limites utiles à défendre en entretien

- Notes utilisées comme labels approximatifs : aucun contrôle humain du sentiment n'est disponible dans la source.
- Corpus alimentaire **anglais, de 1999 à 2012** : aucun résultat revendiqué sur du français, des avis récents ou d'autres catégories.
- L'échantillonnage de textes uniques modifie les proportions et la représentation des produits ; les graphiques décrivent cet échantillon, pas tout Amazon.
- Un texte réutilisé pour plusieurs produits est attribué à sa première occurrence. Les noms des produits sont absents ; seuls les identifiants sont affichés.
- La déduplication ne supprime pas les paraphrases. Des auteurs et produits communs peuvent subsister entre ensembles : le test n'est ni une validation temporelle ni une validation sur des produits inconnus.
- Aucun flux Reddit, rafraîchissement quotidien ou sentiment par aspect. Le transformer est un benchmark expérimental distinct ; la prédiction interactive reste lexicale.
- Les fichiers du modèle sont des artefacts locaux de confiance. Ne pas remplacer un fichier Joblib par un modèle provenant d'une source inconnue.

## Fichiers produits

```text
app.py                              Interface Streamlit
sentiment/data.py                   Nettoyage, labels, lecture SNAP, export
scripts/prepare_data.py             Téléchargement, audit complet, échantillonnage
scripts/train.py                    Comparaison et évaluation reproductibles
data/raw/finefoods.txt.gz           Archive publique originale
data/processed/reviews.parquet      Échantillon réel nettoyé
data/processed/test_predictions.parquet
models/sentiment.joblib             Pipeline TF-IDF + classificateur retenu
reports/data_audit.json             Comptages, règles et empreinte de la source
reports/metrics.json                Performances mesurées
reports/splits.csv                  Affectation des textes aux ensembles
tests/                             Tests des données, artefacts et interactions
```

Les données, modèles et rapports générés volumineux sont ignorés par Git et se recréent avec les scripts. Les noms et identifiants des auteurs ne sont pas conservés dans les fichiers préparés.

## Sources et attribution

- Idée initiale : [Gaël Penessot, Pimp Your Portfolio — analyse de sentiment](https://github.com/gpenessot/pimp-your-portfolio#4-analyse-sentiment-avis-produits).
- Données : [Stanford SNAP, Amazon Fine Foods](https://snap.stanford.edu/data/web-FineFoods.html).
- Référence académique : J. McAuley et J. Leskovec, *From amateurs to connoisseurs: modeling the evolution of user expertise through online reviews*, WWW 2013.

La mise à disposition publique ne confère pas de droit de propriété sur les avis ; conserver l'attribution et vérifier les conditions de la source avant une redistribution ou un usage commercial.
