# Reddit, topics et actualisation quotidienne

## État livré

- Extraction réelle de 8 composantes NMF sur les 39 999 textes Amazon dédupliqués ; une composante dominante par texte pour le décompte. Les trois étoiles sont incluses.
- Collecteur PRAW en lecture seule, testé avec des clients simulés mais **pas validé contre l'API Reddit réelle**, faute d'accès approuvé et d'identifiants.
- Planification GitHub Actions chaque jour à **7 h 15, Europe/Paris**, puis déploiement des agrégats sur la démonstration publique GitHub Pages. Un lancement manuel permet de vérifier le même parcours sans attendre le lendemain.
- Les notes et annotations détaillées existantes restent locales. Aucun texte, titre, auteur, lien de publication ou identifiant de discussion Reddit n'est exporté dans le site.

## Ce que les topics signifient

`scripts/extract_topics.py` ajuste TF-IDF et NMF sur tout l'échantillon Amazon, à visée exploratoire. Il retire les mots vides anglais et une liste explicite de termes génériques d'opinion et d'achat, puis conserve au maximum 8 000 variables. La graine est 42. Le nombre de composantes (8) est un choix d'exploration, pas une estimation validée du nombre réel de thèmes.

Les intitulés sont les trois premiers mots dominants de chaque composante. Le rapport conserve les dix premiers mots, les volumes d'avis et la répartition des notes. Un texte peut être multithématique ; le décompte retient seulement sa composante dominante. La force relative n'est pas une probabilité. La cohérence des thèmes n'a pas été annotée par un humain.

Ces thèmes ne sont **pas** une analyse de sentiment par aspect. Le pipeline positif/négatif et ses évaluations ne sont pas modifiés. Les affectations individuelles sont dans `data/processed/topic_assignments.parquet`, ignoré par Git ; seuls les agrégats sont publiés.

## Activer Reddit ultérieurement

La [FAQ officielle Reddit](https://developers.reddit.com/docs/guides/faq) distingue Devvit de l'accès Data API nécessaire aux scripts externes comme PRAW. Cet accès exige une approbation Reddit ; la possession d'un compte Reddit ne suffit pas. Le projet ne fait aucune démarche d'acceptation de conditions à la place de l'utilisateur.

Une fois l'accès approuvé pour ce cas d'usage :

1. Copier `.streamlit/secrets.example.toml` vers `.streamlit/secrets.toml`.
2. Renseigner `client_id`, `client_secret`, un `user_agent` descriptif avec le vrai compte de contact, et `access_approved = true`. Ne pas publier ce fichier et ne pas coller de secret dans un chat.
3. Adapter `config/reddit.toml` : subreddits publics anglais, recherche, limite et fenêtre temporelle. Par défaut : Coffee et tea, recherche `review OR recommend OR taste`, 100 résultats maximum par subreddit, fenêtre de 7 jours.
4. Lancer `python scripts/collect_reddit.py`, puis `python scripts/refresh_daily.py --require-reddit`.

Les variables d'environnement `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT` et `REDDIT_ACCESS_APPROVED=true` sont une alternative au fichier local.

Pour le workflow GitHub, configurer les trois premières comme **secrets du dépôt** et `REDDIT_ACCESS_APPROVED=true` comme **variable du dépôt**. Les secrets locaux ne sont pas transmis automatiquement à GitHub. À la livraison, aucun de ces accès n'a été configuré ; la collecte demeure inactive comme demandé.

## Collecte et conservation

Le collecteur utilise l'OAuth applicatif PRAW en lecture seule, sans mot de passe de compte, sans publication ni commentaire. Il respecte les limites gérées par PRAW ; il n'utilise aucun contournement, scraping HTML ou accès alternatif si l'API refuse la requête.

Les résultats sont limités à des publications avec corps de texte. Les contenus supprimés, retirés, sans auteur, marqués adultes, épinglés ou hors fenêtre temporelle sont écartés ; les doublons d'identifiant puis de corps normalisé sont retirés. Cette recherche bornée n'est pas exhaustive. Les discussions récupérées ne sont pas automatiquement des avis produits et aucun nom de produit n'est garanti.

Chaque collecte entièrement réussie **remplace** l'instantané local `data/live/reddit.parquet`, sans concaténer l'historique. Ainsi, un post ancien absent d'une nouvelle collecte n'est pas conservé dans le nouvel instantané. Aucun auteur n'est stocké. En cas d'erreur, le dernier fichier local est préservé mais **n'est pas réexporté comme une collecte fraîche** ; le statut public indique l'échec. Cette gestion d'instantanés ne remplace pas les éventuelles obligations supplémentaires de la permission Reddit obtenue.

Le sentiment et les thèmes appliqués à Reddit utilisent les modèles Amazon existants : **aucun apprentissage ou ajustement sur Reddit**, aucune accuracy ou validation revendiquée sur cette source. Les votes Reddit ne servent jamais de labels positif/négatif.

## Actualisation

Le workflow `.github/workflows/daily-refresh.yml` restaure si possible les artefacts Amazon et les modèles, prépare ceux qui manquent, recalcule les agrégats, collecte Reddit seulement si l'accès est configuré et déclaré approuvé, puis redéploie `portfolio/dist/` sur GitHub Pages. Les fichiers Reddit ne sont ni mis en cache dans GitHub Actions ni déposés en artefacts ; seuls leurs agrégats éventuels rejoignent le site public. Les textes ne transitent jamais vers une API d'inférence.

Le classificateur de sentiment est reconstruit depuis les poids JSON déjà publiés, sans réentraînement quotidien. Cela conserve le modèle exact des évaluations existantes et évite de modifier ses prédictions par une nouvelle optimisation sur un autre système. Les thèmes sont recalculés seulement si leurs artefacts manquent ou si l'empreinte du corpus change ; leurs affectations peuvent légèrement varier selon les bibliothèques numériques et la plateforme.

Les fichiers publics `topics.json` et `refresh.json` exposent la méthode, les comptes et les dates utiles. **Un traitement récent ne transforme pas l'archive Amazon en données actuelles** : le dernier avis source reste daté d'octobre 2012. Un état Reddit `not_configured` ne veut pas dire zéro discussion collectée avec succès.

La planification utilise le champ `timezone` pris en charge par [GitHub Actions](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule). Les déclenchements peuvent être retardés par la charge de GitHub ; sur un dépôt public inactif, GitHub peut désactiver les tâches planifiées après 60 jours sans activité. Vérifier la date visible dans le dashboard et les exécutions plutôt que supposer qu'un cron garantit l'heure exacte.

L'actualisation planifiée concerne la **démo publique GitHub Pages**. Elle ne modifie pas automatiquement le dossier de travail local ni la copie privée Sites. Pour actualiser les fichiers locaux :

```powershell
.venv\Scripts\python.exe scripts\refresh_daily.py
```

Le mode `--skip-reddit` désactive explicitement la collecte. `--require-reddit` retourne une erreur si cette collecte n'a pas réussi. Les statuts évitent de présenter un échec ou une absence de configuration comme un succès.
