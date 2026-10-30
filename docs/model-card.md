# Fiche du modèle lexical

## Usage prévu

Classer un texte d'avis en anglais selon la polarité associée aux notes faibles (1–2) ou élevées (4–5) dans le corpus Amazon Fine Foods. Usage de démonstration et d'apprentissage. Les notes sont des labels indirects, pas une annotation humaine du sentiment.

## Modèle et variables

TF-IDF de mots et bigrammes, pondération TF sous-linéaire, normalisation L2, maximum 60 000 variables. Régression logistique C=2, pondération équilibrée des classes. Le corps de l'avis est la seule entrée. Le modèle est choisi sur le macro-F1 de validation face à Naive Bayes, puis sauvegardé sans réentraînement sur le test.

## Données

Source publique Stanford SNAP, archive Amazon Fine Foods jusqu'en octobre 2012. Déduplication du corps normalisé sans distinction de casse ; première occurrence conservée. Tirage uniforme de 40 000 textes uniques, puis retrait d'un conflit binaire présent dans l'échantillon. 39 999 avis restants, dont 37 003 utilisés pour les expériences binaires.

Les identifiants et noms des auteurs ne sont pas conservés. Les noms de produits ne sont pas disponibles. Certaines références du corpus sont hors alimentation, comme observé dans la revue qualitative. Le décompte dans la revue n'est pas une estimation de leur fréquence globale.

## Évaluation

Test initial de 5 551 avis : accuracy 0,9265 et macro-F1 0,8688. Rappel négatif 0,8461 et précision négative 0,7266. Le tableau complet et les protocoles supplémentaires figurent dans le README et les JSON de `reports/`.

## Limites

- Pas de validation en français ou sur un corpus récent.
- Deux classes forcées : avis mixtes et sentiments neutres ne sont pas reconnus comme classes séparées.
- Les scores de la régression ne sont pas calibrés et ne doivent pas être présentés comme une certitude.
- Le test initial n'exclut pas les produits communs ou les paraphrases ; le test par ASIN inconnu traite une partie de cette question, sans regrouper les familles de produits.
- Les expériences temporelle et par produit inconnu sont des stress tests rétrospectifs sur le même échantillon, avec recouvrement des populations de la V1. Ce ne sont pas des validations externes indépendantes.
- Les observations qualitatives sont assistées par IA et attendent une validation humaine.
- Le test initial a été relu : un futur réglage fondé sur ces erreurs exigera un nouveau test.

## Déploiement

Application Streamlit locale avec pipeline Joblib de confiance. Démo web avec les mêmes poids, idf et vocabulaire exportés en JSON : l'inférence s'exécute dans le navigateur sans transmission du texte à une API. La parité des probabilités est vérifiée sur tout le test et des cas de bord par `scripts/check_export.py`.

La démo exporte le modèle lexical. DistilBERT est une comparaison expérimentale ; il n'est pas chargé dans le site ou dans la prédiction interactive Streamlit.
