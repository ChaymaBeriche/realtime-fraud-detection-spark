# Phase 6 — Deployment (détection en temps réel)

Objectif : appliquer le modèle retenu (Random Forest, seuil 0,125) à un **flux continu**
de transactions avec **Spark Structured Streaming**, et lever une alerte pour chaque
transaction suspecte dès son arrivée.

## Architecture

```
01_stream_simulator.py                02_stream_detection.py (Spark Structured Streaming)
rejoue le test, en accéléré   ──►   data/stream/input/   ──►  variables (hour, log_amount)
1 fichier CSV par seconde           (dossier surveillé)         → pipeline Random Forest
(sans la vraie classe)                                          → alerte si score ≥ 0,125
        │                                                       ├─► requête A : transactions notées + alertes
        │                                                       └─► requête B : fenêtres d'1 minute (watermark)
        └─► labels.csv (vraies classes, à part)                           │
                                                                          ▼
                                      03_stream_report.py : complétude, cohérence avec le lot,
                                      qualité des alertes, latence, graphiques
```

## Lancement (avec Docker, deux terminaux)

**Terminal 1** — la détection (laisse le port 4040 pour l'interface Spark) :
```powershell
docker compose run --rm --service-ports spark python 6_deployment/02_stream_detection.py
```
**Terminal 2** — le simulateur, une fois que le terminal 1 affiche « Détection prête » :
```powershell
docker compose run --rm spark python 6_deployment/01_stream_simulator.py
```
Le rejeu dure environ 7 minutes (vitesse x60). Pour une démonstration rapide :
`... 01_stream_simulator.py --speed 600` (~45 secondes).

Les deux scripts peuvent être lancés dans n'importe quel ordre : le simulateur attend que
la détection soit prête. La détection s'arrête seule à la fin du flux. Ensuite :
```powershell
docker compose run --rm spark python 6_deployment/03_stream_report.py
```

## Choix et justifications

| Choix | Pourquoi |
|---|---|
| Source = **dossier surveillé** | Aucun composant supplémentaire ; même code de détection qu'avec Kafka (extension prévue) |
| Fichier écrit sous un nom caché puis **renommé** | Spark ne lit jamais un fichier à moitié écrit |
| La vraie classe **n'est pas dans le flux** | Comme dans la réalité : la fraude n'est confirmée que plus tard ; elle sert seulement au contrôle |
| Mêmes variables qu'en phase 3, **même pipeline** MLlib, même seuil | Le temps réel doit donner exactement les mêmes décisions que l'évaluation |
| Déclenchement toutes les **2 secondes**, `foreachBatch` | Petits lots fréquents : faible latence, écriture des résultats et affichage des alertes |
| **Fenêtres d'1 minute** avec watermark de 2 minutes | Agrégation en temps réel sur l'heure des transactions (et non l'heure d'arrivée) |
| **Checkpoints** | Après une panne, Spark reprend sans perdre ni dupliquer de transactions |
| Fichiers du flux dans un **volume Docker** | Accès rapide aux petits fichiers (liste du dossier, checkpoints) ; sur le dossier Windows partagé, Spark prenait du retard |
| Date de départ conventionnelle (1er septembre 2013) | La source ne donne pas le jour exact ; seule l'heure relative compte |

## Résultats (détail : `outputs/stream_report.md`)

Exécution à la vitesse x60 (7,2 heures de transactions rejouées en 7,2 minutes) :

| Contrôle | Résultat |
|---|---|
| Transactions envoyées / traitées | 52 906 / 52 906 (0 manquante, 0 en double) |
| Écart entre score streaming et score en lot | 0 (identiques pour les 52 906 transactions) |
| Alertes | 70 : **51 vraies fraudes**, 19 fausses alertes ; 19 fraudes manquées |
| Précision / rappel | 0,729 / 0,729 — exactement le résultat de la phase 5 |
| Micro-batches | 217 |
| Latence médiane / 95e centile / max | 1,8 s / 2,0 s / 2,8 s |
| Fenêtres d'1 minute émises | 430 (sur 433 minutes : les dernières attendent le watermark) |

- **La chaîne temps réel est validée de bout en bout** : elle prend exactement les mêmes
  décisions que l'évaluation en lot, transaction par transaction.
- Une alerte est levée en moins de 2 secondes en général après l'arrivée de la transaction ;
  l'essentiel de ce délai vient du déclenchement toutes les 2 secondes.

**Effet du volume Docker** (même PC Windows, même flux) :

| Fichiers du flux | Micro-batches | Latence médiane | 95e centile | Maximum |
|---|---|---|---|---|
| Dossier Windows partagé | 111 | 4,1 s | 9,1 s | 16,7 s |
| Volume Docker | 217 | 1,8 s | 2,0 s | 2,8 s |

Sur le dossier Windows partagé, chaque cycle de traitement prenait ~3,7 s, plus que le
déclenchement de 2 s : Spark prenait du retard et la latence augmentait pendant le flux
(jusqu'à ~15 s à la fin). Les accès aux nombreux petits fichiers (liste du dossier,
checkpoints) sont lents à travers le partage Windows ↔ Linux ; dans un volume Docker,
Spark suit le rythme et la latence reste stable.
- Le débit observé (~120 transactions/s) est imposé par la vitesse de rejeu ; avec
  `--speed 600`, Spark a traité ~1 200 transactions/s avec une latence médiane de 2,5 s.

## Fichiers produits
- `data/stream/` : flux d'entrée, résultats et checkpoints (régénérés à chaque exécution).
  Avec Docker, ce dossier est un **volume Docker** (`stream-data`, voir `docker-compose.yml`) et
  n'apparaît pas dans le dossier Windows : le partage de dossier Windows ↔ Linux est lent pour
  les nombreux petits fichiers du streaming. Les rapports et graphiques, eux, sont bien écrits
  dans `6_deployment/outputs/`.
- `outputs/stream_report.md` — rapport
- `outputs/alerts.csv` — toutes les alertes, avec le verdict (fraude ou fausse alerte)
- `outputs/figures/stream_timeline.png` — transactions, alertes et vraies fraudes par minute
- `outputs/figures/stream_latency.png` — latence par micro-batch
