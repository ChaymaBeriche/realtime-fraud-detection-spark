"""
Phase 6 — Deployment
Script 03 : rapport du flux temps réel et contrôle de bout en bout.

À lancer après les scripts 01 et 02. Il vérifie que la chaîne temps réel donne
exactement les mêmes résultats que l'évaluation en lot de la phase 5 :
  1. complétude : toutes les transactions envoyées ont-elles été traitées, une seule fois ?
  2. cohérence : pour chaque transaction, le score calculé en streaming est-il identique
     au score calculé en lot par le même modèle ?
  3. qualité des alertes, comparées aux vraies classes (gardées à part, labels.csv)
  4. latence : temps entre l'envoi d'une transaction et son traitement par Spark
  5. fenêtres d'une minute produites par la requête de streaming

Entrées : data/stream/output/ (script 02), data/stream/labels.csv (script 01),
          data/prepared/test, models/decision.json
Sorties : 6_deployment/outputs/stream_report.md
          6_deployment/outputs/alerts.csv
          6_deployment/outputs/figures/stream_timeline.png
          6_deployment/outputs/figures/stream_latency.png

Usage (depuis la racine du projet) :
    python 6_deployment/03_stream_report.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array

import config as cfg

OUTPUT_DIR = cfg.DEPLOYMENT_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
C_BLUE, C_ORANGE = "#2a78d6", "#eb6834"
LABELS = {"logistic_regression": "Régression logistique", "random_forest": "Random Forest", "gbt": "GBT"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2)


def main():
    decision = json.loads(cfg.DECISION_FILE.read_text(encoding="utf-8"))
    threshold = float(decision["threshold"])
    spark = cfg.spark_session("FraudDetection-StreamReport")

    # ------------------------------------------------------------ résultats du streaming
    streamed = pd.read_parquet(cfg.STREAM_OUTPUT_DIR / "scored")
    batches = pd.read_csv(cfg.STREAM_OUTPUT_DIR / "batches.csv")
    windows_path = cfg.STREAM_OUTPUT_DIR / "windows"
    windows = (pd.read_parquet(windows_path).sort_values("window_start")
               if windows_path.exists() and any(windows_path.glob("*.parquet")) else pd.DataFrame())
    labels = pd.read_csv(cfg.STREAM_LABELS_FILE)

    # ------------------------------------------------------------ 1. complétude
    n_sent = len(labels)
    n_rows = len(streamed)
    n_unique = streamed["transaction_id"].nunique()
    missing = n_sent - len(set(labels["transaction_id"]) & set(streamed["transaction_id"]))
    duplicates = n_rows - n_unique

    # ------------------------------------------------------------ 2. cohérence avec le calcul en lot
    # mêmes identifiants que le simulateur : test trié par Time, numéroté dans cet ordre
    test = pd.read_parquet(cfg.PREPARED_DIR / "test").sort_values("Time", kind="mergesort")
    test = test.reset_index(drop=True)
    test.insert(0, "transaction_id", range(len(test)))
    model = PipelineModel.load(str(cfg.ROOT / decision["model_path"]))
    batch_scores = (model.transform(spark.createDataFrame(test))
                         .select("transaction_id", vector_to_array("probability")[1].alias("batch_score"))
                         .toPandas())
    both = streamed.drop_duplicates("transaction_id").merge(batch_scores, on="transaction_id")
    max_diff = float((both["score"] - both["batch_score"]).abs().max())
    batch_alerts = int((batch_scores["batch_score"] >= threshold).sum())
    same_decisions = int(((both["score"] >= threshold) == (both["batch_score"] >= threshold)).sum())

    # ------------------------------------------------------------ 3. qualité des alertes
    res = streamed.drop_duplicates("transaction_id").merge(labels, on="transaction_id")
    fraud = res[cfg.LABEL_COL] == 1
    alert = res["alert"]
    tp, fp, fn = int((alert & fraud).sum()), int((alert & ~fraud).sum()), int((~alert & fraud).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    alerts = (res[alert].sort_values("event_time")
              .assign(verdict=lambda d: np.where(d[cfg.LABEL_COL] == 1, "fraude", "fausse alerte"))
              [["transaction_id", "event_time", "Amount", "score", "verdict"]])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    alerts.to_csv(OUTPUT_DIR / "alerts.csv", index=False, float_format="%.4f")

    # ------------------------------------------------------------ 4. latence et débit
    latency = res["processed_at"] - res["sent_at"]
    duration = res["processed_at"].max() - res["sent_at"].min()
    throughput = n_unique / duration if duration > 0 else float("nan")
    data_hours = (res["Time"].max() - res["Time"].min()) / 3600

    # ------------------------------------------------------------ 5. graphiques
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    per_min = (res.assign(minute=res["event_time"].dt.floor("min"))
                  .groupby("minute")
                  .agg(transactions=("transaction_id", "size"),
                       alerts=("alert", "sum"),
                       frauds=(cfg.LABEL_COL, "sum")))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    a1.bar(per_min.index, per_min["transactions"], width=1 / 1440, color=C_BLUE)
    style(a1, "Transactions reçues par minute", "", "transactions")
    a2.bar(per_min.index, per_min["alerts"], width=1 / 1440, color=C_ORANGE, label="alertes levées")
    true_f = per_min[per_min["frauds"] > 0]
    a2.scatter(true_f.index, true_f["frauds"], s=18, color=INK, zorder=3, label="vraies fraudes")
    style(a2, "Alertes levées et vraies fraudes par minute", "heure des transactions", "nombre")
    a2.legend(frameon=False, loc="upper left")
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%Hh%M"))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "stream_timeline.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(batches["batch_id"], batches["latency_median_s"], color=C_BLUE, linewidth=2,
            marker="o", markersize=4, label="médiane")
    ax.plot(batches["batch_id"], batches["latency_max_s"], color=C_ORANGE, linewidth=2,
            marker="o", markersize=4, label="maximum")
    style(ax, "Latence par micro-batch (envoi → traitement)", "micro-batch", "secondes")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "stream_latency.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------ rapport
    complete = missing == 0 and duplicates == 0
    consistent = max_diff < 1e-9 and same_decisions == len(both)
    lines = [
        "# Phase 6 — Rapport du flux temps réel\n",
        f"Modèle : {LABELS.get(decision['model'], decision['model'])}, seuil {threshold:.4f}. Flux : les {n_sent:,} transactions du "
        f"test ({data_hours:.1f} h de données), rejouées en {duration / 60:.1f} minutes.\n",
        "## 1. Complétude\n",
        "| Transactions envoyées | Traitées | Manquantes | En double |", "|---|---|---|---|",
        f"| {n_sent:,} | {n_unique:,} | {missing} | {duplicates} |",
        "",
        ("**Toutes les transactions ont été traitées, une seule fois.**" if complete else
         "**Attention : des transactions manquent ou sont en double** (voir tableau)."), "",
        "## 2. Cohérence avec l'évaluation en lot (phase 5)\n",
        "Chaque transaction a été notée une seconde fois, en lot, par le même modèle.\n",
        "| Contrôle | Résultat |", "|---|---|",
        f"| Écart maximal entre score streaming et score en lot | {max_diff:.2e} |",
        f"| Transactions avec la même décision (alerte ou non) | {same_decisions:,} / {len(both):,} |",
        f"| Alertes en lot / en streaming | {batch_alerts} / {int(alert.sum())} |",
        "",
        ("**La chaîne temps réel donne exactement les mêmes résultats que l'évaluation en lot.**"
         if consistent else "**Écart détecté entre streaming et lot : à analyser.**"), "",
        "## 3. Qualité des alertes (vraies classes gardées à part)\n",
        "| Alertes | Vraies fraudes détectées | Fausses alertes | Fraudes manquées | Précision | Rappel |",
        "|---|---|---|---|---|---|",
        f"| {int(alert.sum())} | {tp} | {fp} | {fn} | {precision:.3f} | {recall:.3f} |",
        "",
        "La liste complète des alertes (avec le verdict) est dans `alerts.csv`.\n",
        "## 4. Latence et débit\n",
        "| Mesure | Valeur |", "|---|---|",
        f"| Micro-batches traités | {len(batches)} |",
        f"| Latence médiane (envoi → traitement) | {latency.median():.1f} s |",
        f"| Latence au 95e centile | {latency.quantile(0.95):.1f} s |",
        f"| Latence maximale | {latency.max():.1f} s |",
        f"| Débit moyen | {throughput:.0f} transactions / seconde (fixé par la vitesse de rejeu, "
        "pas la capacité maximale de Spark) |",
        "",
        f"La latence comprend l'attente du prochain déclenchement (toutes les "
        f"{cfg.STREAM_TRIGGER.replace('seconds', 'secondes')}), "
        "la détection du fichier et le calcul. Elle concerne le traitement d'une transaction, "
        "pas la vitesse de rejeu.\n",
        "## 5. Fenêtres d'une minute (requête de streaming avec watermark)\n",
    ]
    if len(windows):
        n_minutes = per_min.shape[0]
        lines += [
            f"- {len(windows)} fenêtres émises, pour {n_minutes} minutes de transactions.",
            f"- Transactions comptées dans les fenêtres : {int(windows['transactions'].sum()):,} ; "
            f"alertes : {int(windows['alerts'].sum())}.",
            f"- Fenêtre la plus chargée : {int(windows['transactions'].max())} transactions ; "
            f"maximum d'alertes dans une minute : {int(windows['alerts'].max())}.",
            "- Avec un watermark de 2 minutes, une fenêtre n'est émise que lorsque des transactions "
            "plus récentes de 2 minutes sont arrivées : les dernières minutes du flux ne sont donc "
            "jamais émises (aucune donnée ne vient après elles). C'est le comportement normal "
            "d'une agrégation en mode « append ».",
            "",
        ]
    else:
        lines += ["Aucune fenêtre émise.", ""]
    lines += [
        "## Graphiques",
        "- `figures/stream_timeline.png` — transactions, alertes et vraies fraudes par minute",
        "- `figures/stream_latency.png` — latence par micro-batch",
    ]
    text = "\n".join(lines)
    (OUTPUT_DIR / "stream_report.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    spark.stop()


if __name__ == "__main__":
    main()
