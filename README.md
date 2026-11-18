# Olist Seller Insights

> 🚧 En construction. Ce README sera complété à la fin du MVP (étape 8).

Outil d'analyse de la performance des vendeurs d'une marketplace, à partir du dataset public
[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) :

1. **Pipeline de données** Bronze / Silver / Gold dans DuckDB ;
2. **Modèle de ML** qui repère les vendeurs dont la satisfaction client risque de chuter ;
3. **Analyse par LLM** des avis clients (traduits PT → EN), avec une sortie JSON vérifiée automatiquement.

## Problème

_À compléter._

## Architecture

_À compléter (schéma)._

## Lancer le projet

Prérequis : Linux ou WSL2, Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env          # puis renseigner les clés API (étape 5)
pytest
```

Données : télécharger le dataset Olist sur Kaggle et placer les CSV dans `data/raw/` (non commité).

## Choix techniques

_À compléter._

## Résultats

_À compléter._

## Limites

_À compléter._

## Licence des données

Dataset Olist : CC BY-NC-SA 4.0.
