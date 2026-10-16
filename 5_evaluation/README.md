# Phase 5 — Evaluation

Objectif : choisir le seuil de décision du modèle retenu (Random Forest), puis mesurer
une seule fois ses performances finales sur le test, jamais utilisé jusqu'ici.

## Scripts (à lancer depuis la racine du projet, dans cet ordre)

| Script | Rôle | Produit |
|---|---|---|
| `01_choose_threshold.py` | Choisit le seuil qui maximise le F2 sur la **validation** | `models/decision.json`, `outputs/threshold_report.md`, `figures/threshold_validation.png` |
| `02_evaluate_test.py` | Évaluation finale sur le **test** avec ce seuil | `outputs/test_report.md`, `figures/test_pr_curve.png`, `figures/test_scores.png` |
| `common.py` | Fonctions partagées (scores, métriques, graphiques) | — |

```powershell
docker compose run --rm --service-ports spark python 5_evaluation/01_choose_threshold.py
docker compose run --rm --service-ports spark python 5_evaluation/02_evaluate_test.py
```

## Choix et justifications

| Choix | Pourquoi |
|---|---|
| Seuil choisi en **maximisant le F2** | Le F2 compte le rappel deux fois plus que la précision : une fraude manquée coûte plus cher qu'une vérification inutile. Aucune valeur arbitraire (coût d'une alerte, devise) n'intervient |
| Seuil choisi sur la **validation**, jamais sur le test | Le test doit rester une mesure honnête de la performance future |
| Seuil placé au milieu entre deux scores | Garde une marge pour des transactions nouvelles |
| Montants de fraude présentés à titre d'information | Vue métier concrète, sans influencer le choix |
| Intervalles de confiance par **bootstrap** (1 000 rééchantillonnages) | Avec 70 fraudes, un chiffre seul serait trompeur |

## Résultats

**Seuil retenu : 0,125** (détail : `outputs/threshold_report.md`). Le F2 reste à moins de
5 % de son maximum pour tous les seuils entre 0,05 et 0,40 : le choix est stable.

**Test** (détail : `outputs/test_report.md`) — 52 906 transactions, 70 fraudes :

| | Valeur | Intervalle à 95 % |
|---|---|---|
| Fraudes détectées | 51 / 70 | — |
| Fausses alertes | 19 | — |
| Précision | 0,729 | [0,617 ; 0,829] |
| Rappel | 0,729 | [0,623 ; 0,822] |
| F2 | 0,729 | [0,637 ; 0,810] |
| AUC-PR | 0,764 | — |
| AUC-ROC | 0,983 | — |
| Montant de fraude détecté | 68 % (5 219 sur 7 629) | [39 % ; 94 %] |

- Environ **10 alertes par heure**, dont 73 % sont de vraies fraudes.
- Les résultats du test sont proches de ceux de la validation (rappel 0,729 contre 0,754) :
  le seuil se généralise bien.
- **Fraudes manquées** : 19, avec un score médian de 0,003 contre 0,762 pour les fraudes
  détectées. Pour le modèle, elles ressemblent à des transactions normales. Les rattraper
  coûterait très cher : avec un seuil de 0,01, on détecterait 4 fraudes de plus mais avec
  232 fausses alertes au lieu de 19 ; avec 0,005, 9 de plus pour 798 fausses alertes
  (vérification faite après coup, sans changer le seuil retenu). Parmi elles figurent
  des montants élevés (1 097, 634, 358), d'où l'intervalle large sur la part du montant détecté.
- **Fausses alertes** : 19, surtout de très petits montants (médiane 1,00).
- Pour information, la régression logistique (AUC-PR 0,759) et le GBT (0,742) sont proches
  du Random Forest sur le test : les écarts entre modèles sont du même ordre que l'incertitude.

## Limites
- Le test ne contient que 70 fraudes : d'où les intervalles de confiance larges.
- Le test couvre les heures relatives 16 à 23 : la détection de nuit n'est pas mesurée.

## Utilisé ensuite par
- Phase 6 : `models/decision.json` (Random Forest + seuil 0,125) pour lever les alertes
  dans le flux en temps réel
