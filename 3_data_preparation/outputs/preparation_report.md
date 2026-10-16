# Phase 3 — Rapport de préparation

## 1. Suppression des doublons

| | Lignes | Fraudes |
|---|---|---|
| Avant | 284,807 | 492 |
| Après | 283,726 | 473 |
| Supprimées | 1,081 | 19 |

## 2. Nouvelles variables

- `hour` = floor(Time / 3600) mod 24 : heure relative (0–23)
- `log_amount` = log(1 + Amount) : montant en échelle logarithmique
- `Time` est conservé uniquement pour l'ordre des transactions (pas comme variable de modèle)

Variables d'entrée des modèles (30) : V1…V28, log_amount, hour.

## 3. Découpage temporel

Durée totale : 48.0 h. Coupures à 33.6 h (70%) et 40.8 h (85%).

| Partie | Période | Transactions | Fraudes | Taux de fraude | Part des fraudes |
|---|---|---|---|---|---|
| train | 0.0 h → 33.6 h | 171,446 | 342 | 0.199 % | 72.3 % |
| validation | 33.6 h → 40.8 h | 59,374 | 61 | 0.103 % | 12.9 % |
| test | 40.8 h → 48.0 h | 52,906 | 70 | 0.132 % | 14.8 % |

Les parties ne se chevauchent pas dans le temps : le modèle est toujours évalué sur des transactions postérieures à celles de l'entraînement.

## 4. Fichiers produits

Un fichier Parquet par partie, lignes triées sur toutes les colonnes (ordre identique sur toutes les machines).

- `data/prepared/train/`
- `data/prepared/validation/`
- `data/prepared/test/`

Colonnes : Time, Amount, V1, V2, …, log_amount, hour, Class (33 au total).
