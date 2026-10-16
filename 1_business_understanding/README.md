# Phase 1 — Business Understanding

## Problème
Les banques et émetteurs de cartes doivent repérer les transactions frauduleuses
**au moment où elles arrivent**, dans un flux continu, pour pouvoir les bloquer ou
les vérifier avant que la perte ne soit définitive. Les transactions sont trop
nombreuses pour une vérification manuelle, et la fraude est très rare (moins de
0,2 % des transactions dans nos données), ce qui la rend difficile à repérer.

## Objectif du projet
Construire une chaîne de détection de fraude **en temps réel** avec Apache Spark
(Python / PySpark) :
1. entraîner des modèles de Machine Learning (Spark MLlib) sur des transactions
   historiques étiquetées ;
2. appliquer le meilleur modèle à un flux continu de transactions
   (Spark Structured Streaming) et lever une alerte pour chaque transaction suspecte ;
3. compléter la détection supervisée par une détection d'anomalies non supervisée,
   pour repérer des comportements inhabituels.

## Données
Dataset réel **Credit Card Fraud Detection** (ULB / Worldline, Kaggle) :
284 807 transactions de porteurs de cartes européens sur 2 jours, dont 492 fraudes.
Détails en phase 2.

## Critères de réussite
| Critère | Pourquoi |
|---|---|
| **AUC-PR** (aire sous la courbe précision-rappel) | Métrique principale : adaptée aux classes très déséquilibrées, contrairement à l'accuracy ou à l'AUC-ROC |
| **Rappel** (part des fraudes détectées) | Une fraude manquée coûte cher |
| **Précision** (part des alertes qui sont de vraies fraudes) | Trop de fausses alertes surcharge les analystes et gêne les clients |
| **Latence** du traitement en streaming | Une alerte n'est utile que si elle arrive vite |

Le compromis rappel / précision sera fixé en phase 5 en choisissant le seuil de décision.

## Contraintes
- Les variables V1–V28 sont anonymisées (ACP) : on ne peut pas les interpréter métier.
- Seulement 2 jours de données : la validation doit respecter l'ordre du temps
  (entraîner sur le passé, tester sur le futur).
