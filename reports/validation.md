# Vérification de la V1 — 30 octobre 2026

- Archive SNAP intégralement parcourue : 568 454 enregistrements valides selon les contrôles du script, sans champ requis manquant. L'empreinte du fichier téléchargé est conservée dans `data_audit.json`.
- 39 999 textes dans l'échantillon final, dont 37 003 binaires. Les hashes des textes sont uniques et ne se croisent pas entre entraînement, validation et test.
- Régression logistique choisie sur validation ; test séparé de 5 551 avis. Accuracy 0,92650 ; macro-F1 0,86881 ; référence majoritaire 0,84435 et 0,45780 respectivement.
- `python -m pytest -q` : **7 tests réussis en 31,95 s**. Tests exécutés hors bac à sable, les processus Streamlit de test restant bloqués dans le bac à sable Windows. Les processus bloqués lancés pour la vérification ont été arrêtés.
- Tests : politique des labels et nettoyage ; lecture des blocs SNAP et de l'encodage ; protection des exports CSV ; unicité des splits ; prédictions sauvegardées et recomputation de l'accuracy ; navigation et cas sans résultat ; avis vide et prédictions positive/négative ; pages évaluation et méthode.
- Contrôle visuel dans le navigateur intégré à `http://127.0.0.1:8501/` : titre, navigation, filtres, quatre métriques, barres de répartition des notes et barres du volume annuel rendus. Les valeurs visibles correspondent à l'audit. Le comportement responsive observé dans le panneau étroit empile les métriques et les graphiques. Aucun contrôle visuel exhaustif sur plusieurs tailles d'écran n'a été effectué.
- Application démarrée uniquement sur l'adresse de bouclage `127.0.0.1`, sans publication externe.

Limites analytiques visibles dans l'application et le README : labels indirects issus des notes, corpus anglais historique, proportions de textes uniques, absence d'évaluation temporelle et de sentiment par aspect. Les métriques mesurent ce protocole précis.

## Évaluations et publication complémentaires

- Deux stress tests exécutés avec le pipeline figé : macro-F1 temporel 0,87396 (5 571 avis) ; macro-F1 ASIN inconnus 0,86029 (5 441 avis). Les manifests vérifient les dates et groupes disjoints. Ces expériences recoupent les populations de la V1 et ne sont pas des validations externes.
- 80 corps d'avis lus avec assistance IA : 20 par cellule de confusion, 40 erreurs et 40 prédictions conformes aux notes. Annotations détaillées maintenues **locales**, selon le choix explicite de l'utilisateur ; synthèse publique. Hashes contrôlés avant jointure pour éviter la réattribution d'une annotation à un autre texte.
- Benchmark CPU sur le même test de 5 551 avis : DistilBERT SST-2 macro-F1 0,77173 contre 0,86881 pour la régression ; 313 avis tronqués à 256 tokens. Révision figée, safetensors et aucun code distant exécuté.
- Bootstrap apparié : différence transformer − lexical −0,09708, intervalle à 95 % [−0,11135 ; −0,08211]. Hypothèse d'indépendance par avis ; dépendances auteurs/produits non corrigées.
- Export navigateur vérifié sur 5 555 textes, sans différence de classe ; écart maximal de probabilité 6,66 × 10⁻¹⁶.
- Suite locale complète : **11 tests réussis** après génération des artefacts. Syntaxe Python et JavaScript vérifiée. Les tests CI sans artefacts sautent les cas concernés explicitement.
- Contrôle visuel de Streamlit enrichi sur le port temporaire 8503 : graphique de robustesse, valeurs du tableau et revue assistée affichés. Captures enregistrées dans `docs/screenshots/`.
- Démonstration navigateur vérifiée localement : exemples positif et négatif correctement classés et absence d'appel d'inférence externe. Les champs de texte utilisent un affichage texte et les exports neutralisent les formules CSV.
- Dépôt public créé sur GitHub ; GitHub Pages déployé. La première CI a révélé une attente erronée sur un fichier local, corrigée par un saut explicite. Le pipeline complet Linux a ensuite révélé une assertion de navigation exigeant une carte métrique sur une vue tabulaire, également corrigée. La vérification finale du pipeline complet est suivie dans GitHub Actions.
- Dockerfile fourni et documenté, non construit ici : Docker n'est pas installé.

## État final vérifié

- Suite locale : 11 tests réussis en 16,39 s après la correction de navigation.
- CI GitHub sur le code final : [réussie](https://github.com/Late-In-Data/Avis-produit/actions/runs/41009484672).
- Pipeline complet Linux sur ce même code, avec téléchargement SNAP, entraînement et stress tests : [réussi](https://github.com/Late-In-Data/Avis-produit/actions/runs/41009508178). Les tests d'annotations détaillées locales et de prédictions transformer locales y sont sautés explicitement ; ils sont exécutés dans la suite locale.
- [Déploiement GitHub Pages réussi](https://github.com/Late-In-Data/Avis-produit/actions/runs/41009484693). Le site public affiche les cinq lignes du tableau de résultats et les métriques finales du transformer.
- Prédictions contrôlées sur le site public : exemple positif à 98,1 % de score positif ; exemple négatif à 0,1 %. Capture de la page des résultats sauvegardée dans `docs/screenshots/demo-publique.jpg`.
- Copie Sites déployée avec succès, accès privé au propriétaire conservé. Les annotations détaillées ne figurent dans aucun des déploiements ou dans les fichiers suivis par Git.
