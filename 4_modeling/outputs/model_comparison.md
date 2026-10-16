# Phase 4 — Comparaison des modèles supervisés

Entraînement : 171,446 transactions dont 342 fraudes. Évaluation : validation (61 fraudes).

Poids de classe (variante « avec poids ») : normal = 0.501, fraude = 250.7.

## Tous les essais (triés par AUC-PR)

| Modèle | Réglage | Poids | AUC-PR | AUC-ROC | Durée |
|---|---|---|---|---|---|
| Random Forest | numTrees=100, maxDepth=5 | sans poids | 0.7506 | 0.9679 | 18 s |
| Random Forest | numTrees=100, maxDepth=10 | avec poids | 0.7377 | 0.9602 | 40 s |
| GBT | maxIter=100, maxDepth=3 | sans poids | 0.7074 | 0.9536 | 38 s |
| Random Forest | numTrees=100, maxDepth=5 | avec poids | 0.6999 | 0.9794 | 19 s |
| GBT | maxIter=50, maxDepth=3 | sans poids | 0.6909 | 0.9735 | 21 s |
| Random Forest | numTrees=100, maxDepth=10 | sans poids | 0.6520 | 0.9760 | 41 s |
| Régression logistique | regParam=0.01 | sans poids | 0.6212 | 0.9713 | 5 s |
| Régression logistique | regParam=0.1 | sans poids | 0.6190 | 0.9761 | 6 s |
| Régression logistique | regParam=0.0 | avec poids | 0.5893 | 0.9733 | 9 s |
| Régression logistique | regParam=0.1 | avec poids | 0.5638 | 0.9616 | 5 s |
| GBT | maxIter=100, maxDepth=5 | sans poids | 0.5533 | 0.9491 | 52 s |
| GBT | maxIter=50, maxDepth=5 | sans poids | 0.5491 | 0.9331 | 28 s |
| Régression logistique | regParam=0.01 | avec poids | 0.5217 | 0.9678 | 6 s |
| Régression logistique | regParam=0.0 | sans poids | 0.4876 | 0.9704 | 13 s |
| GBT | maxIter=100, maxDepth=3 | avec poids | 0.4832 | 0.9806 | 38 s |
| GBT | maxIter=50, maxDepth=3 | avec poids | 0.4561 | 0.9802 | 21 s |
| GBT | maxIter=50, maxDepth=5 | avec poids | 0.4328 | 0.9277 | 28 s |
| GBT | maxIter=100, maxDepth=5 | avec poids | 0.4108 | 0.9226 | 54 s |

## Effet des poids de classe (meilleur réglage de chaque modèle)

| Modèle | AUC-PR sans poids | AUC-PR avec poids |
|---|---|---|
| Régression logistique | 0.6212 | 0.5893 |
| Random Forest | 0.7506 | 0.7377 |
| GBT | 0.7074 | 0.4832 |

Sur ces données, les poids de classe ne changent presque pas l'AUC-PR pour : Random Forest ; dégradent l'AUC-PR pour : Régression logistique, GBT (seuil : 5 % d'écart relatif).

## Meilleur réglage par modèle (sauvegardé dans `models/`)

| Modèle | Réglage | Poids | AUC-PR | AUC-ROC |
|---|---|---|---|---|
| Random Forest | numTrees=100, maxDepth=5 | sans poids | 0.7506 | 0.9679 |
| GBT | maxIter=100, maxDepth=3 | sans poids | 0.7074 | 0.9536 |
| Régression logistique | regParam=0.01 | sans poids | 0.6212 | 0.9713 |

## Variables les plus importantes (Random Forest)

V17 (0.133), V12 (0.123), V7 (0.108), V11 (0.090), V9 (0.086), V16 (0.082), V4 (0.080), V10 (0.069), V14 (0.046), V18 (0.035)

## Remarque
La validation ne contient que 61 fraudes : un écart de quelques centièmes d'AUC-PR entre deux modèles n'est pas significatif.

## Graphiques
- `figures/pr_curves_validation.png` — courbes précision-rappel
- `figures/feature_importance.png` — importance des variables
