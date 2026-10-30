# Revue qualitative assistée de 80 avis

Cette revue a été réalisée par Codex avec assistance IA après lecture intégrale des 80 corps d'avis sélectionnés. **Ce n'est pas une annotation humaine indépendante**. Les notes et les prédictions étaient visibles pendant la lecture.

## Sélection et traçabilité

`scripts/prepare_review.py` sélectionne aléatoirement 20 avis dans chacune des quatre cellules de la matrice de confusion du test initial, puis mélange l'ordre. Graine de sélection : 2026 ; graine de mélange : 2027. Cela donne 40 erreurs et 40 prédictions conformes à la note. Les erreurs sont volontairement surreprésentées : aucun pourcentage issu de ces 80 avis ne doit être extrapolé au corpus.

Chaque avis possède un identifiant R001–R080 et un hash SHA-256 du texte normalisé. Les annotations et justifications sont conservées localement dans `reports/assisted_annotations.csv`, accessible dans l'application Streamlit. Ce fichier détaillé n'est pas publié par défaut. Les textes se régénèrent depuis la source publique avec les scripts ; ils ne sont pas republiés dans le dépôt. La synthèse et sa méthode sont publiques.

La lecture assistée attribue un sentiment global parmi positif, négatif, mixte ou indéterminé, une catégorie explicative principale et une justification courte. Ce vocabulaire sert à l'analyse des erreurs : **le modèle de production reste binaire**. Une réserve mineure ne rend pas automatiquement un avis mixte : on retient l'appréciation dominante lorsqu'elle est explicite.

## Observations dans cette sélection

- 40 avis lus comme négatifs, 33 comme positifs, 6 comme mixtes et 1 comme indéterminé.
- 8 des 40 erreurs ont pour catégorie principale des aspects opposés : goût apprécié mais prix excessif, défaut de texture ou emballage défaillant.
- 3 erreurs sont principalement liées à des négations favorables, par exemple l'absence d'un défaut ; 3 concernent une expérience qui évolue d'un problème initial à une satisfaction finale.
- R031 et R071 ont un texte négatif malgré une note de 4 ou 5 étoiles. Ce sont des **désaccords possibles**, qui demandent validation humaine ; ils ne prouvent pas que toutes les erreurs de modèle sont des erreurs de labels.
- 10 exemples hors alimentation ont été repérés : accessoires de capsules, sacs, bonsaï, pièges, aluminium, jouet et cosmétiques. Le nom « Fine Foods » ne suffit pas à garantir le périmètre réel de tous les avis.
- Les comparaisons avec d'autres produits et les avis d'autres clients cités dans le texte brouillent aussi l'identification de la cible du sentiment.

Ces comptes proviennent de `scripts/summarize_review.py`. Les catégories sont subjectives et exclusives pour le décompte, alors que plusieurs phénomènes peuvent coexister dans un même avis. Les affirmations sanitaires ou nutritionnelles contenues dans certains avis ne sont pas validées par cette lecture.

## Décision pour la prochaine version

Conserver le sentiment binaire et rendre les limites visibles. Une analyse par aspect pourrait aider certains cas, mais cette revue ne constitue pas un jeu d'évaluation permettant d'en mesurer la qualité. Avant d'optimiser, définir le périmètre (aliment lui-même ou expérience d'achat), faire annoter des exemples en aveugle, puis réserver un nouveau test.

Le formulaire `reports/human_review_template.csv` se génère avec les scripts et s'exporte dans la page « Analyse approfondie ». Il ne montre initialement ni note ni prédiction. Aucun champ humain n'a été rempli à la place de l'utilisateur. Une double annotation humaine et un arbitrage des désaccords permettront une référence plus solide.

Le test initial ayant maintenant été inspecté, toute adaptation fondée sur cette revue doit être évaluée sur un nouveau jeu réservé. Les stress tests de ce projet réutilisent le même corpus et ne remplacent pas une validation externe.
