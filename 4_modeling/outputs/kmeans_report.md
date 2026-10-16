# Phase 4 — Détection d'anomalies avec KMeans

KMeans est appris sur le train **sans les étiquettes**. Les étiquettes servent uniquement à mesurer le résultat et à décrire les groupes obtenus.

## 1. Choix de k (silhouette)

| k | Silhouette |
|---|---|
| 5 | 0.0746 |
| 10 ← retenu | 0.0990 |
| 20 | 0.0871 |

## 2. Groupes obtenus (k = 10)

8 grands groupes (≥ 1% du train) et 2 petits groupes. Composition des petits groupes (étiquettes utilisées ici pour décrire, pas pour apprendre) :

| Groupe | Transactions | dont fraudes |
|---|---|---|
| 9 | 216 | 176 |
| 8 | 910 | 3 |

Les petits groupes contiennent 179 des 342 fraudes du train, pour 1,126 transactions sur 171,446.

## 3. Deux scores d'anomalie (validation)

| Score | Définition | AUC-PR | AUC-ROC |
|---|---|---|---|
| A | distance au centre de son groupe | 0.0292 | 0.9520 |
| B (CBLOF) | distance au grand groupe le plus proche | 0.0273 | 0.9514 |

Idée du score B : si des fraudes forment leur propre petit groupe, elles sont proches du centre de ce groupe et le score A les manque ; le score B mesure plutôt leur distance aux comportements les plus courants (grands groupes).

Les scores A et B sont **comparables** sur cette validation (moins de 10 % d'écart d'AUC-PR).

## 4. Comparaison avec le meilleur modèle supervisé (Random Forest)

Même nombre d'alertes pour chaque méthode. Validation : 59,374 transactions, 61 fraudes.

| Budget d'alertes | Score A | Score B | Random Forest | B seulement | B ou Random Forest |
|---|---|---|---|---|---|
| 0.5% (297 alertes) | 0 | 1 | 50 | 0 | 50 |
| 2.0% (1187 alertes) | 43 | 42 | 52 | 0 | 52 |

Chaque case donne le nombre de fraudes trouvées parmi les alertes.

## 5. Conclusion (calculée à partir des résultats ci-dessus)
- KMeans range **52% des fraudes** dans des petits groupes qui ne représentent que 0.7% des transactions, sans avoir vu les étiquettes : les fraudes forment des groupes à part.
- Silhouette faible (< 0,25) pour toutes les valeurs de k : les transactions ne forment pas de groupes nettement séparés, et le choix de k est fragile.
- Parmi les 297 transactions les plus inhabituelles selon le score B, 1 fraude(s) seulement (0.3%). Leur montant médian est de 171.32, contre 28.00 pour l'ensemble de la validation. Inhabituel ne veut donc pas dire frauduleux.
- Avec 297 alertes, KMeans (score B) trouve moins de fraudes que le Random Forest (1 contre 50) et trouve 0 fraude(s) que le Random Forest manque.
- Avec 1187 alertes, KMeans (score B) trouve moins de fraudes que le Random Forest (42 contre 52) et trouve 0 fraude(s) que le Random Forest manque.
- KMeans n'apporte donc **pas de complément mesurable** au Random Forest sur ces données.

## Graphique
- `figures/kmeans_scores.png` — distribution du score B (normal / fraude)
