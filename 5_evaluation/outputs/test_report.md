# Phase 5 — Évaluation finale sur le test

Modèle : **Random Forest**, seuil **0.1251** (choisi sur la validation par maximisation du F2).
Test : 52,906 transactions sur 7.2 heures, 70 fraudes. Le test n'a servi à aucun choix.

## 1. Matrice de confusion

| | Prédit fraude (alerte) | Prédit normal |
|---|---|---|
| **Vraie fraude** | 51 (détectées) | 19 (manquées) |
| **Vraie normale** | 19 (fausses alertes) | 52,817 |

## 2. Métriques (intervalle de confiance à 95 %, bootstrap 1000 rééchantillonnages)

| Métrique | Valeur | Intervalle à 95 % |
|---|---|---|
| Précision | 0.729 | [0.617 ; 0.829] |
| Rappel | 0.729 | [0.623 ; 0.822] |
| F1 | 0.729 | — |
| F2 | 0.729 | [0.637 ; 0.810] |
| AUC-PR | 0.764 | — |
| AUC-ROC | 0.983 | — |

Avec 70 fraudes seulement, le rappel réel se situe vraisemblablement entre 62% et 82%.

## 3. Vue métier

- 70 alertes sur 52,906 transactions (0.13%), soit environ 10 par heure ; 73% d'entre elles sont de vraies fraudes.
- Montant des fraudes : 7 628.68 au total, **5 218.84 détecté (68%)**, 2 409.84 manqué. Intervalle à 95 % de la part détectée : [39% ; 94%].

## 4. Validation et test

| | Alertes | Fraudes détectées | Fausses alertes | Précision | Rappel | F2 |
|---|---|---|---|---|---|---|
| Validation | 60 | 46 | 14 | 0.767 | 0.754 | 0.757 |
| Test | 70 | 51 | 19 | 0.729 | 0.729 | 0.729 |

Le rappel sur le test est **proche** de celui de la validation : le seuil se généralise bien.

## 5. Analyse des erreurs

| Groupe | Nombre | Montant médian | Score médian | Part de montants nuls |
|---|---|---|---|---|
| Fraudes détectées | 51 | 3.90 | 0.762 | 8% |
| Fraudes manquées | 19 | 8.00 | 0.003 | 5% |
| Fausses alertes | 19 | 1.00 | 0.201 | 0% |
| Transactions normales (toutes) | 52,836 | 17.98 | 0.000 | 0% |

Les 19 fraudes manquées (triées par score) :

| Montant | Heure relative | Score |
|---|---|---|
| 0.76 | 18 | 0.102 |
| 9.29 | 17 | 0.078 |
| 1.00 | 17 | 0.047 |
| 1.00 | 17 | 0.016 |
| 187.11 | 19 | 0.010 |
| 12.31 | 21 | 0.009 |
| 7.06 | 19 | 0.006 |
| 3.14 | 19 | 0.005 |
| 17.39 | 17 | 0.005 |
| 8.00 | 16 | 0.003 |
| 19.95 | 22 | 0.002 |
| 357.95 | 18 | 0.002 |
| 42.53 | 23 | 0.002 |
| 0.00 | 22 | 0.001 |
| 7.59 | 19 | 0.001 |
| 2.47 | 18 | 0.001 |
| 1096.99 | 18 | 0.000 |
| 1.00 | 20 | 0.000 |
| 634.30 | 22 | 0.000 |

Les fraudes manquées ont un score médian de 0.003, contre 0.762 pour les fraudes détectées : pour le modèle, elles ressemblent à des transactions normales. Leur montant médian (8.00) est plus élevé que celui des fraudes détectées (3.90).

Rappel : le test couvre les heures relatives 16 à 23 seulement (aucune heure de nuit, voir phase 3).

## 6. Pour information : autres modèles sur le test

Le choix du modèle a été fait sur la validation ; ces chiffres ne servent pas à le modifier.

| Modèle | AUC-PR test | AUC-ROC test |
|---|---|---|
| Random Forest (retenu) | 0.764 | 0.983 |
| Régression logistique | 0.759 | 0.976 |
| GBT | 0.742 | 0.942 |

## Graphiques
- `figures/test_pr_curve.png` — courbe précision-rappel et point de fonctionnement
- `figures/test_scores.png` — distribution des scores (normal / fraude) et seuil
