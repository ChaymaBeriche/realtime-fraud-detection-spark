# Phase 2 — Data Understanding

Objectif : vérifier que les données sont complètes et fiables, puis comprendre
ce qui distingue une fraude d'une transaction normale.

## Scripts (à lancer depuis la racine du projet, dans cet ordre)

| Script | Rôle | Produit |
|---|---|---|
| `01_load_and_validate.py` | Lit le CSV avec un schéma explicite, vérifie lignes / fraudes / valeurs manquantes / doublons, puis sauvegarde une copie Parquet | `outputs/validation_report.md`, `data/parquet/creditcard/` |
| `02_eda.py` | Analyse exploratoire avec Spark SQL et graphiques | `outputs/eda_summary.md`, `outputs/figures/*.png` |

```bash
python 2_data_understanding/01_load_and_validate.py
python 2_data_understanding/02_eda.py
```

## Résultats principaux

**Validation** (détail : `outputs/validation_report.md`)
- 284 807 transactions, 492 fraudes (0,173 %) : identique au dataset officiel.
- Aucune valeur manquante ; 1 081 lignes en double (traitées en phase 3).

**Exploration** (détail : `outputs/eda_summary.md`)
- **Déséquilibre** : 1 fraude pour 577 transactions. Prédire toujours « normal »
  donnerait 99,83 % d'accuracy : on utilisera AUC-PR, rappel et précision.
- **Montants** (`figures/01_montants.png`) : médiane plus faible pour les fraudes
  (9,21 contre 22,00) mais 95e centile plus élevé (648 contre 364) ; 5,5 % des
  fraudes ont un montant nul. Le montant seul ne suffit pas.
- **Heure** (`figures/02_heures.png`) : le taux de fraude atteint 1,71 % à
  l'heure relative 2, quand le volume est faible (~10 fois la moyenne).
- **Variables** (`figures/03_correlations.png`, `figures/04_top_variables.png`) :
  V17, V14, V12 et V10 séparent le mieux les fraudes (moyenne autour de −6 contre ~0).
- **Doublons** : 1 081 copies, dont 19 fraudes.

## Décisions pour la phase 3
- Supprimer les doublons exacts **avant** de séparer entraînement et test.
- Ajouter la variable `hour` (heure relative).
- Mettre `Amount` à l'échelle (les V1–V28 le sont déjà par l'ACP).
- Séparer entraînement / test **dans l'ordre du temps**.
