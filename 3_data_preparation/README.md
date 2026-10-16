# Phase 3 — Data Preparation

Objectif : transformer les données validées en phase 2 en trois jeux propres
(entraînement, validation, test), prêts pour la modélisation.

## Script (à lancer depuis la racine du projet)

| Script | Rôle | Produit |
|---|---|---|
| `01_prepare.py` | Supprime les doublons, crée `hour` et `log_amount`, découpe les données dans l'ordre du temps | `data/prepared/{train,validation,test}/`, `outputs/preparation_report.md` |

```powershell
docker compose run --rm --service-ports spark python 3_data_preparation/01_prepare.py
```

## Choix et justifications

| Choix | Pourquoi |
|---|---|
| Supprimer les 1 081 doublons exacts **avant** le découpage | Sinon une même transaction pourrait se trouver à la fois dans l'entraînement et dans le test |
| `hour = floor(Time / 3600) mod 24` à la place de `Time` | `Time` ne fait que croître ; l'heure relative garde le signal utile (plus de fraude la nuit) |
| `log_amount = log(1 + Amount)` | Les montants vont de 0 à plus de 25 000 ; le log les ramène à une échelle comparable |
| Standardisation **en phase 4**, dans le pipeline MLlib | Elle doit être apprise sur le train uniquement, sans regarder la validation ni le test |
| Découpage **temporel** 70 / 15 / 15 % de la durée | Simule un vrai déploiement : on apprend sur le passé et on évalue sur le futur |
| Lignes triées sur toutes les colonnes, un fichier par partie | Ordre identique sur toutes les machines : les modèles de la phase 4 donnent les mêmes résultats partout |

## Résultats (détail : `outputs/preparation_report.md`)

- 284 807 → 283 726 transactions (1 081 doublons supprimés, dont 19 fraudes) ; 473 fraudes restantes.

| Partie | Période | Transactions | Fraudes | Taux |
|---|---|---|---|---|
| train | 0 → 33,6 h | 171 446 | 342 | 0,199 % |
| validation | 33,6 → 40,8 h | 59 374 | 61 | 0,103 % |
| test | 40,8 → 48 h | 52 906 | 70 | 0,132 % |

## Limite à garder en tête
Avec seulement 2 jours de données, la validation (heures relatives 9 à 16) et le
test (heures 16 à 23) ne contiennent **aucune heure de nuit** : les heures 0 à 8,
où le taux de fraude est le plus élevé, ne sont présentes que dans l'entraînement.
Les scores de validation et de test mesurent donc la détection en journée et en
soirée. C'est le prix d'un découpage réaliste dans le temps.

## Utilisé ensuite par
- Phase 4 : `train` pour l'entraînement, `validation` pour comparer les modèles
- Phase 5 : `validation` pour choisir le seuil, `test` pour le score final
- Phase 6 : `test` rejoué comme flux de transactions en temps réel
