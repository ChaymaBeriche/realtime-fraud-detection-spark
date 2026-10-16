# Phase 6 — Rapport du flux temps réel

Modèle : Random Forest, seuil 0.1251. Flux : les 52,906 transactions du test (7.2 h de données), rejouées en 7.2 minutes.

## 1. Complétude

| Transactions envoyées | Traitées | Manquantes | En double |
|---|---|---|---|
| 52,906 | 52,906 | 0 | 0 |

**Toutes les transactions ont été traitées, une seule fois.**

## 2. Cohérence avec l'évaluation en lot (phase 5)

Chaque transaction a été notée une seconde fois, en lot, par le même modèle.

| Contrôle | Résultat |
|---|---|
| Écart maximal entre score streaming et score en lot | 0.00e+00 |
| Transactions avec la même décision (alerte ou non) | 52,906 / 52,906 |
| Alertes en lot / en streaming | 70 / 70 |

**La chaîne temps réel donne exactement les mêmes résultats que l'évaluation en lot.**

## 3. Qualité des alertes (vraies classes gardées à part)

| Alertes | Vraies fraudes détectées | Fausses alertes | Fraudes manquées | Précision | Rappel |
|---|---|---|---|---|---|
| 70 | 51 | 19 | 19 | 0.729 | 0.729 |

La liste complète des alertes (avec le verdict) est dans `alerts.csv`.

## 4. Latence et débit

| Mesure | Valeur |
|---|---|
| Micro-batches traités | 217 |
| Latence médiane (envoi → traitement) | 1.8 s |
| Latence au 95e centile | 2.0 s |
| Latence maximale | 2.8 s |
| Débit moyen | 122 transactions / seconde (fixé par la vitesse de rejeu, pas la capacité maximale de Spark) |

La latence comprend l'attente du prochain déclenchement (toutes les 2 secondes), la détection du fichier et le calcul. Elle concerne le traitement d'une transaction, pas la vitesse de rejeu.

## 5. Fenêtres d'une minute (requête de streaming avec watermark)

- 430 fenêtres émises, pour 433 minutes de transactions.
- Transactions comptées dans les fenêtres : 52,725 ; alertes : 70.
- Fenêtre la plus chargée : 243 transactions ; maximum d'alertes dans une minute : 3.
- Avec un watermark de 2 minutes, une fenêtre n'est émise que lorsque des transactions plus récentes de 2 minutes sont arrivées : les dernières minutes du flux ne sont donc jamais émises (aucune donnée ne vient après elles). C'est le comportement normal d'une agrégation en mode « append ».

## Graphiques
- `figures/stream_timeline.png` — transactions, alertes et vraies fraudes par minute
- `figures/stream_latency.png` — latence par micro-batch
