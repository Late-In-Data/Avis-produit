# Vérification de la V1 — 30 octobre 2026

- Archive SNAP intégralement parcourue : 568 454 enregistrements valides selon les contrôles du script, sans champ requis manquant. L'empreinte du fichier téléchargé est conservée dans `data_audit.json`.
- 39 999 textes dans l'échantillon final, dont 37 003 binaires. Les hashes des textes sont uniques et ne se croisent pas entre entraînement, validation et test.
- Régression logistique choisie sur validation ; test séparé de 5 551 avis. Accuracy 0,92650 ; macro-F1 0,86881 ; référence majoritaire 0,84435 et 0,45780 respectivement.
- `python -m pytest -q` : **7 tests réussis en 31,95 s**. Tests exécutés hors bac à sable, les processus Streamlit de test restant bloqués dans le bac à sable Windows. Les processus bloqués lancés pour la vérification ont été arrêtés.
- Tests : politique des labels et nettoyage ; lecture des blocs SNAP et de l'encodage ; protection des exports CSV ; unicité des splits ; prédictions sauvegardées et recomputation de l'accuracy ; navigation et cas sans résultat ; avis vide et prédictions positive/négative ; pages évaluation et méthode.
- Contrôle visuel dans le navigateur intégré à `http://127.0.0.1:8501/` : titre, navigation, filtres, quatre métriques, barres de répartition des notes et barres du volume annuel rendus. Les valeurs visibles correspondent à l'audit. Le comportement responsive observé dans le panneau étroit empile les métriques et les graphiques. Aucun contrôle visuel exhaustif sur plusieurs tailles d'écran n'a été effectué.
- Application démarrée uniquement sur l'adresse de bouclage `127.0.0.1`, sans publication externe.

Limites analytiques visibles dans l'application et le README : labels indirects issus des notes, corpus anglais historique, proportions de textes uniques, absence d'évaluation temporelle et de sentiment par aspect. Les métriques mesurent ce protocole précis.
