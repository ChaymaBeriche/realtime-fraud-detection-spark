# Phase 5 — Choix du seuil de décision (validation)

Modèle : **Random Forest** (retenu en phase 4). Validation : 59,374 transactions, 61 fraudes.

Règle : maximiser le F2, qui compte le rappel deux fois plus que la précision.

## 1. Seuil retenu

**Seuil = 0.1251** : une alerte est levée si la probabilité de fraude ≥ 0.1251.

| Alertes | Fraudes détectées | Fausses alertes | Fraudes manquées | Précision | Rappel | F1 | F2 |
|---|---|---|---|---|---|---|---|
| 60 | 46 | 14 | 15 | 0.767 | 0.754 | 0.760 | 0.757 |

## 2. Stabilité du choix

Le F2 reste ≥ 95% de son maximum pour les seuils entre **0.053** et **0.398** (zone grisée du graphique). Le choix est **stable** : le F2 reste proche du maximum sur une plage large de seuils.

## 3. Compromis précision / rappel selon le seuil

| Seuil | Alertes | Fraudes détectées | Fausses alertes | Précision | Rappel | F2 | Montant détecté | Montant manqué |
|---|---|---|---|---|---|---|---|---|
| 0.050 | 83 | 47 | 36 | 0.566 | 0.770 | 0.719 | 7 853.77 | 4 476.31 |
| 0.100 | 67 | 46 | 21 | 0.687 | 0.754 | 0.740 | 7 852.77 | 4 477.31 |
| 0.200 | 56 | 45 | 11 | 0.804 | 0.738 | 0.750 | 7 852.00 | 4 478.08 |
| 0.300 | 48 | 44 | 4 | 0.917 | 0.721 | 0.753 | 7 852.00 | 4 478.08 |
| 0.400 | 44 | 41 | 3 | 0.932 | 0.672 | 0.712 | 7 486.41 | 4 843.67 |
| 0.500 | 41 | 39 | 2 | 0.951 | 0.639 | 0.684 | 7 368.06 | 4 962.02 |
| 0.600 | 32 | 30 | 2 | 0.938 | 0.492 | 0.543 | 4 668.85 | 7 661.23 |
| 0.700 | 28 | 26 | 2 | 0.929 | 0.426 | 0.478 | 4 299.43 | 8 030.65 |
| 0.800 | 16 | 16 | 0 | 1.000 | 0.262 | 0.308 | 2 637.52 | 9 692.56 |
| 0.900 | 0 | 0 | 0 | 0.000 | 0.000 | 0.000 | 0.00 | 12 330.08 |
| 0.125 **(retenu)** | 60 | 46 | 14 | 0.767 | 0.754 | 0.757 | 7 852.77 | 4 477.31 |

Montant total des fraudes de la validation : 12 330.08 (devise non précisée par la source).

## 4. Vue métier (au seuil retenu)

- 60 alertes à vérifier sur 59,374 transactions (0.10%), soit environ 8 par heure (la validation couvre 7.2 heures).
- 7 852.77 de fraude détectée, 4 477.31 manquée (soit 64% du montant total des fraudes).

## Fichiers
- `models/decision.json` — modèle et seuil retenus (utilisés en phase 6)
- `figures/threshold_validation.png` — précision, rappel et F2 selon le seuil
