# Phase 4 — Modeling

Objectif : entraîner des modèles de détection de fraude avec Spark MLlib, les
comparer sur la validation, et tester une détection d'anomalies non supervisée.

## Scripts (à lancer depuis la racine du projet, dans cet ordre)

| Script | Rôle | Produit |
|---|---|---|
| `01_train_supervised.py` | Régression logistique, Random Forest, GBT ; avec et sans poids de classe ; petite grille de réglages | `models/{logistic_regression,random_forest,gbt}/`, `outputs/model_comparison.md`, `outputs/supervised_results.csv`, `outputs/best_models.json`, 2 graphiques |
| `02_train_kmeans.py` | KMeans non supervisé et score d'anomalie ; comparaison avec le meilleur modèle supervisé | `models/kmeans/`, `models/kmeans_meta.json`, `outputs/kmeans_report.md`, 1 graphique |

```powershell
docker compose run --rm --service-ports spark python 4_modeling/01_train_supervised.py
docker compose run --rm --service-ports spark python 4_modeling/02_train_kmeans.py
```
Durée indicative : ~10 min pour le script 01, ~2 min pour le script 02.

## Choix et justifications

| Choix | Pourquoi |
|---|---|
| Pipeline MLlib `VectorAssembler → StandardScaler → modèle` | La standardisation est apprise sur le train seulement, et le pipeline complet est réutilisé tel quel en streaming (phase 6) |
| Poids de classe `N / (2 × n_classe)` : ~0,5 (normal), ~251 (fraude) | Chaque classe pèse autant à l'entraînement ; comparé à la version sans poids |
| Réglages évalués sur la **validation**, pas de `CrossValidator` | `CrossValidator` mélange les lignes au hasard et casserait le découpage temporel |
| AUC-PR comme métrique de choix | Adaptée aux classes très déséquilibrées (l'AUC-ROC reste élevée même pour un modèle médiocre) |
| KMeans : k choisi par silhouette, sans étiquettes | Méthode réellement non supervisée |

## Résultats — modèles supervisés (détail : `outputs/model_comparison.md`)

| Modèle (meilleur réglage) | AUC-PR | AUC-ROC |
|---|---|---|
| **Random Forest** (100 arbres, profondeur 5, sans poids) | **0,751** | 0,968 |
| GBT (100 itérations, profondeur 3, sans poids) | 0,707 | 0,954 |
| Régression logistique (regParam 0,01, sans poids) | 0,621 | 0,971 |

- Le **Random Forest** est le meilleur sur la validation ; c'est lui qui sera utilisé ensuite.
- **Les poids de classe n'améliorent l'AUC-PR pour aucun modèle** : effet quasi nul pour le
  Random Forest, dégradation pour la régression logistique et surtout pour le GBT.
  Explication : l'AUC-PR mesure le classement des transactions, et les poids changent
  surtout l'échelle des probabilités. Un poids de 251 pousse aussi le GBT à sur-apprendre
  les 342 fraudes du train. Le déséquilibre sera traité en phase 5 par le **choix du seuil**.
- Variables les plus importantes : V17, V12, V7, V11, V9 — cohérent avec la phase 2.
- Avec 61 fraudes en validation, quelques centièmes d'écart ne sont pas significatifs.

## Résultats — KMeans (détail : `outputs/kmeans_report.md`)

- k = 10 (meilleure silhouette), mais la silhouette est faible pour toutes les valeurs
  testées (< 0,1) : les transactions ne forment pas de groupes nettement séparés.
- Sans voir les étiquettes, KMeans range tout de même **52 % des fraudes du train** dans
  de petits groupes qui ne représentent que 0,7 % des transactions (dont un groupe de
  216 transactions contenant 176 fraudes).
- Deux scores d'anomalie ont été comparés : distance au centre de son groupe (A) et
  distance au grand groupe le plus proche (B, méthode CBLOF). Ils sont comparables
  (AUC-PR ~0,03, contre ~0,75 pour le Random Forest).
- Sur 297 alertes, KMeans trouve 1 fraude contre 50 pour le Random Forest ; sur 1 187
  alertes, 42 contre 52. Il ne trouve **aucune** fraude que le Random Forest manque.
- Les transactions les plus « inhabituelles » sont surtout des transactions normales de
  gros montant (médiane 171 contre 28 en général) : inhabituel ne veut pas dire frauduleux.

## Reproductibilité
Random Forest, GBT et KMeans font des tirages aléatoires qui dépendent du découpage des
données en partitions Spark. Pour obtenir les mêmes résultats sur toutes les machines :
Spark utilise toujours 4 threads (`SPARK_THREADS` dans `config.py`) et la phase 3 écrit
des fichiers triés, un par partie. Deux exécutions complètes donnent des résultats identiques.

## Utilisé ensuite par
- Phase 5 : `models/random_forest` (et les autres pour comparaison) — choix du seuil sur
  la validation, score final sur le test
- Phase 6 : le modèle retenu est chargé par Spark Structured Streaming
