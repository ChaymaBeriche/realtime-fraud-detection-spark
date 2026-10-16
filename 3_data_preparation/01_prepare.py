"""
Phase 3 — Data Preparation
Script 01 : nettoyage, nouvelles variables et découpage temporel.

Étapes :
  1. suppression des doublons exacts (1 081 lignes, vus en phase 2)
  2. nouvelles variables :
       hour       = floor(Time / 3600) mod 24   heure relative de la journée
       log_amount = log(1 + Amount)             montant ramené à une échelle comparable
  3. découpage dans l'ordre du temps (on entraîne sur le passé, on teste sur le futur) :
       train      = 70 % premiers de la durée totale
       validation = 15 % suivants
       test       = 15 % derniers (rejoué comme flux en phase 6)
  4. sauvegarde des trois parties en Parquet + rapport

La mise à l'échelle (standardisation) n'est PAS faite ici : elle sera apprise
sur le train uniquement, à l'intérieur du pipeline MLlib de la phase 4.

Entrée  : data/parquet/creditcard        (phase 2)
Sorties : data/prepared/{train,validation,test}/
          3_data_preparation/outputs/preparation_report.md

Usage (depuis la racine du projet) :
    python 3_data_preparation/01_prepare.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyspark.sql import functions as F

import config as cfg

OUTPUT_DIR = cfg.DATA_PREPARATION_DIR / "outputs"


def main():
    spark = cfg.spark_session("FraudDetection-Prepare")
    raw = spark.read.parquet(str(cfg.PARQUET_DIR))
    n_raw = raw.count()
    n_raw_fraud = raw.filter(F.col(cfg.LABEL_COL) == 1).count()

    # ------------------------------------------------------------ 1. doublons
    df = raw.dropDuplicates()
    n_clean = df.count()
    n_clean_fraud = df.filter(F.col(cfg.LABEL_COL) == 1).count()

    # ------------------------------------------------------------ 2. variables
    df = (df.withColumn("hour", (F.floor(F.col("Time") / 3600) % 24).cast("int"))
            .withColumn("log_amount", F.log1p(F.col("Amount"))))

    # ------------------------------------------------------------ 3. découpage temporel
    t_max = df.agg(F.max("Time")).first()[0]
    t_train_end = cfg.TRAIN_END_FRACTION * t_max
    t_val_end = cfg.VALIDATION_END_FRACTION * t_max

    df = df.withColumn(
        "split",
        F.when(F.col("Time") < t_train_end, "train")
         .when(F.col("Time") < t_val_end, "validation")
         .otherwise("test"),
    )
    df.cache()

    # colonnes conservées : Time (pour l'ordre du flux en phase 6), Amount (montant
    # lisible pour les rapports), les variables des modèles et la classe
    keep = ["Time", "Amount"] + cfg.MODEL_FEATURES + [cfg.LABEL_COL]

    # ------------------------------------------------------------ 4. sauvegarde
    stats = (df.groupBy("split")
               .agg(F.count("*").alias("rows"),
                    F.sum(cfg.LABEL_COL).alias("frauds"),
                    F.min("Time").alias("t_min"),
                    F.max("Time").alias("t_max"))
               .toPandas().set_index("split"))

    # Reproductibilité : tri sur TOUTES les colonnes (ordre unique, puisque les doublons
    # sont supprimés) et un seul fichier par partie. Les lignes sont ainsi toujours dans
    # le même ordre, quelle que soit la machine, et les modèles de la phase 4 voient
    # exactement les mêmes données dans le même ordre.
    for split in cfg.SPLITS:
        (df.filter(F.col("split") == split).select(keep)
           .orderBy(*keep)
           .coalesce(1)
           .write.mode("overwrite").parquet(str(cfg.PREPARED_DIR / split)))

    # ------------------------------------------------------------ rapport
    def hours(seconds):
        return f"{seconds / 3600:.1f} h"

    report = [
        "# Phase 3 — Rapport de préparation\n",
        "## 1. Suppression des doublons\n",
        "| | Lignes | Fraudes |", "|---|---|---|",
        f"| Avant | {n_raw:,} | {n_raw_fraud} |",
        f"| Après | {n_clean:,} | {n_clean_fraud} |",
        f"| Supprimées | {n_raw - n_clean:,} | {n_raw_fraud - n_clean_fraud} |\n",
        "## 2. Nouvelles variables\n",
        "- `hour` = floor(Time / 3600) mod 24 : heure relative (0–23)",
        "- `log_amount` = log(1 + Amount) : montant en échelle logarithmique",
        "- `Time` est conservé uniquement pour l'ordre des transactions (pas comme variable de modèle)\n",
        f"Variables d'entrée des modèles ({len(cfg.MODEL_FEATURES)}) : "
        f"V1…V28, log_amount, hour.\n",
        "## 3. Découpage temporel\n",
        f"Durée totale : {hours(t_max)}. Coupures à {hours(t_train_end)} "
        f"({cfg.TRAIN_END_FRACTION:.0%}) et {hours(t_val_end)} ({cfg.VALIDATION_END_FRACTION:.0%}).\n",
        "| Partie | Période | Transactions | Fraudes | Taux de fraude | Part des fraudes |",
        "|---|---|---|---|---|---|",
    ]
    for split in cfg.SPLITS:
        r = stats.loc[split]
        report.append(
            f"| {split} | {hours(r.t_min)} → {hours(r.t_max)} | {int(r.rows):,} | "
            f"{int(r.frauds)} | {100 * r.frauds / r.rows:.3f} % | "
            f"{100 * r.frauds / n_clean_fraud:.1f} % |")
    report += [
        "",
        "Les parties ne se chevauchent pas dans le temps : le modèle est toujours "
        "évalué sur des transactions postérieures à celles de l'entraînement.\n",
        "## 4. Fichiers produits\n",
        "Un fichier Parquet par partie, lignes triées sur toutes les colonnes "
        "(ordre identique sur toutes les machines).\n",
        "- `data/prepared/train/`",
        "- `data/prepared/validation/`",
        "- `data/prepared/test/`",
        "",
        f"Colonnes : {', '.join(keep[:4])}, …, {', '.join(keep[-3:])} ({len(keep)} au total).",
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    text = "\n".join(report)
    (OUTPUT_DIR / "preparation_report.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    spark.stop()


if __name__ == "__main__":
    main()
