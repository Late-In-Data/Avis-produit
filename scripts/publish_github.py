"""Publication explicite du projet sur le compte GitHub déjà authentifié.

Aucun secret n'est écrit dans le dépôt ou dans la sortie. Le gestionnaire Git
existant est utilisé sans ouvrir de nouvelle authentification interactive.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OWNER = "Late-In-Data"
NAME = "Avis-produit"
REMOTE = f"https://github.com/{OWNER}/{NAME}.git"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true", help="Créer le dépôt public si absent et pousser les fichiers suivis")
    parser.add_argument("--homepage", default="")
    args = parser.parse_args()
    if not args.publish:
        parser.error("Action externe explicite : ajouter --publish pour publier.")
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="Never")
    credential = subprocess.run(["git", "credential", "fill"], input="protocol=https\nhost=github.com\n\n",
                                text=True, capture_output=True, timeout=20, check=True, env=env)
    fields = dict(line.split("=", 1) for line in credential.stdout.splitlines() if "=" in line)
    token = fields.get("password")
    if not token:
        raise RuntimeError("Aucune authentification GitHub enregistrée.")

    def request(path, method="GET", data=None):
        headers = {"Authorization": f"Bearer {token}", "User-Agent": "Avis-produit-portfolio",
                   "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        body = json.dumps(data).encode() if data is not None else None
        if body is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request("https://api.github.com"+path, headers=headers, method=method, data=body)
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)

    profile = request("/user")
    if profile["login"].casefold() != OWNER.casefold():
        raise RuntimeError("Le compte authentifié ne correspond pas au propriétaire prévu.")
    try:
        repo = request(f"/repos/{OWNER}/{NAME}")
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        repo = request("/user/repos", "POST", {"name": NAME, "private": False,
                       "description": "Sentiment binaire d’avis Amazon : audit, évaluation temporelle et produits inconnus, comparaison transformer, application Streamlit et démo navigateur.",
                       "homepage": args.homepage, "auto_init": False})
    if repo["owner"]["login"].casefold() != OWNER.casefold():
        raise RuntimeError("Propriétaire du dépôt inattendu.")
    if not (ROOT / ".git").exists():
        subprocess.run(["git", "init", "--initial-branch", "main"], cwd=ROOT, check=True)
    existing = subprocess.run(["git", "remote", "get-url", "origin"], cwd=ROOT, capture_output=True, text=True)
    if existing.returncode == 0 and existing.stdout.strip() != REMOTE:
        raise RuntimeError("Une autre origine Git est déjà configurée ; aucun remplacement automatique.")
    if existing.returncode != 0:
        subprocess.run(["git", "remote", "add", "origin", REMOTE], cwd=ROOT, check=True)
    subprocess.run(["git", "add", "--all"], cwd=ROOT, check=True)
    staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT)
    if staged.returncode == 1:
        subprocess.run(["git", "-c", "user.name="+OWNER,
                        "-c", f"user.email={profile['id']}+{OWNER}@users.noreply.github.com",
                        "commit", "-m", "Build reproducible review sentiment portfolio and evaluations"], cwd=ROOT, check=True)
    elif staged.returncode != 0:
        raise RuntimeError("Impossible de vérifier les changements Git.")
    subprocess.run(["git", "push", "-u", "origin", "main"], cwd=ROOT, env=env, check=True)
    # Démonstration publique du dépôt public via GitHub Pages, après création de main.
    try:
        pages = request(f"/repos/{OWNER}/{NAME}/pages")
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        pages = request(f"/repos/{OWNER}/{NAME}/pages", "POST", {"build_type": "workflow"})
    if args.homepage:
        request(f"/repos/{OWNER}/{NAME}", "PATCH", {"homepage": args.homepage})
    request(f"/repos/{OWNER}/{NAME}/topics", "PUT", {"names": ["sentiment-analysis", "nlp", "streamlit", "portfolio", "scikit-learn", "transformers", "data-science"]})
    print(json.dumps({"repository": repo["html_url"], "private": repo["private"], "pages": pages.get("html_url")}))


if __name__ == "__main__":
    main()
