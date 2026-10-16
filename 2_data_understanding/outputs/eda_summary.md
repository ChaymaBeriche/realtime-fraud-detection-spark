# Phase 2 — Analyse exploratoire (EDA)

## 1. Répartition des classes

| Classe | Transactions | Part |
|---|---|---|
| Normal (0) | 284,315 | 99.827 % |
| Fraude (1) | 492 | 0.173 % |

Ratio : **1 fraude pour 577 transactions normales**. L'exactitude (accuracy) est donc inutilisable : un modèle qui prédit toujours « normal » obtiendrait 99.83 %.

## 2. Montants (devise non précisée par la source)

| Classe | Moyenne | Médiane | 95e centile | Max | Montant = 0 |
|---|---|---|---|---|---|
| Normal | 88.29 | 22.0 | 364.0 | 25,691.16 | 0.63 % |
| Fraude | 122.21 | 9.21 | 648.0 | 2,125.87 | 5.49 % |

Les fraudes ont une médiane plus faible mais une moyenne et un 95e centile plus élevés : beaucoup de très petits montants (5,5 % valent 0) et une queue de montants élevés. Le montant seul ne suffit pas à isoler la fraude.

## 3. Évolution selon l'heure

`hour = floor(Time / 3600) mod 24` (heure relative au début de la collecte).

- Taux de fraude maximal à l'heure **2** : 1.71 % (57 fraudes sur 3,328 transactions)
- Taux moyen : 0.173 %
- La nuit (peu de transactions), la part de fraude est bien plus élevée : l'heure est une variable utile.

## 4. Variables les plus liées à la fraude

Corrélation de Pearson avec `Class` et moyenne par classe (top 10).

| Variable | Corrélation | Moyenne normal | Moyenne fraude |
|---|---|---|---|
| V17 | -0.326 | +0.01 | -6.67 |
| V14 | -0.303 | +0.01 | -6.97 |
| V12 | -0.261 | +0.01 | -6.26 |
| V10 | -0.217 | +0.01 | -5.68 |
| V16 | -0.197 | +0.01 | -4.14 |
| V3 | -0.193 | +0.01 | -7.03 |
| V7 | -0.187 | +0.01 | -5.57 |
| V11 | +0.155 | -0.01 | +3.80 |
| V4 | +0.133 | -0.01 | +4.54 |
| V18 | -0.111 | +0.00 | -2.25 |

Variables les moins liées : V26, V15, V25, V23, V22. Les corrélations restent modestes car la classe fraude est minuscule ; les écarts de moyenne, eux, sont nets (V14, V17, V12…). Un modèle non linéaire devrait en tirer parti.

## 5. Doublons

- Lignes en double (copies en trop) : 1,081
  - dont normales : 1,062
  - dont fraudes : 19
- Décision : supprimer les doublons exacts avant le split train/test (phase 3 — Data Preparation), sinon une même transaction pourrait se retrouver des deux côtés.

## Graphiques

- `figures/01_montants.png` — distribution des montants
- `figures/02_heures.png` — volume et taux de fraude par heure
- `figures/03_correlations.png` — corrélations avec la fraude
- `figures/04_top_variables.png` — distributions des variables clés
