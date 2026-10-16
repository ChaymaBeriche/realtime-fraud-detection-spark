"""
Phase 2 — Data Understanding
Script 01 : chargement et validation du dataset réel ULB / Worldline.

Source : Kaggle « Credit Card Fraud Detection » (mlg-ulb/creditcardfraud).
Transactions réelles de porteurs de cartes européens, septembre 2013, sur 2 jours.
Colonnes : Time (secondes depuis la 1re transaction), V1..V28 (composantes ACP
anonymisées), Amount (montant), Class (1 = fraude).

Le script :
  1. lit le CSV avec un schéma explicite (pas d'inférence : plus rapide et plus sûr)
  2. vérifie l'intégrité : nb de lignes, nb de fraudes, valeurs manquantes, doublons
  3. écrit le rapport outputs/validation_report.md
  4. sauvegarde une copie Parquet (data/parquet/creditcard), utilisée par la suite

Le fichier CSV d'origine n'est jamais modifié.

Usage (depuis la racine du projet) :
    python 2_data_understanding/01_load_and_validate.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType

import config as cfg

OUTPUT_DIR = cfg.DATA_UNDERSTANDING_DIR / "outputs"

SCHEMA = StructType(
    [StructField("Time", DoubleType(), False)]
    + [StructField(c, DoubleType(), False) for c in cfg.PCA_COLS]
    + [StructField("Amount", DoubleType(), False),
       StructField("Class", IntegerType(), False)]
)


def main():
    if not cfg.RAW_CSV.exists():
        sys.exit(f"Fichier introuvable : {cfg.RAW_CSV}\n"
                 "Téléchargez-le depuis https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")

    spark = cfg.spark_session("FraudDetection-Load")
    df = spark.read.csv(str(cfg.RAW_CSV), header=True, schema=SCHEMA, mode="FAILFAST")

    n_rows = df.count()
    n_fraud = df.filter(F.col(cfg.LABEL_COL) == 1).count()
    n_null = df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns]) \
               .first().asDict()
    n_dup = n_rows - df.dropDuplicates().count()
    hours = df.agg(F.max("Time")).first()[0] / 3600
    ok = n_rows == cfg.EXPECTED_ROWS and n_fraud == cfg.EXPECTED_FRAUDS

    report = [
        "# Rapport de validation du dataset\n",
        f"Fichier : `data/archive/creditcard.csv` ({cfg.RAW_CSV.stat().st_size:,} octets)\n",
        "| Contrôle | Résultat | Attendu |",
        "|---|---|---|",
        f"| Lignes | {n_rows:,} | {cfg.EXPECTED_ROWS:,} |",
        f"| Fraudes | {n_fraud:,} ({n_fraud / n_rows:.3%}) | {cfg.EXPECTED_FRAUDS} |",
        f"| Colonnes | {len(df.columns)} | 31 |",
        f"| Valeurs manquantes | {sum(n_null.values())} | 0 |",
        f"| Doublons exacts | {n_dup:,} | — (traités en phase 3) |",
        f"| Période couverte | {hours:.1f} heures | ~48 heures |",
        "",
        f"**Verdict : {'conforme au dataset officiel' if ok else 'NON conforme — arrêt'}.**\n",
        "## Schéma utilisé",
        "",
        "```",
        df._jdf.schema().treeString().strip(),
        "```",
        "",
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if ok:
        # 4 fichiers d'environ 16 Mo : lecture parallèle par Spark et fichiers faciles à copier
        df.repartition(4).write.mode("overwrite").parquet(str(cfg.PARQUET_DIR))
        report.append("Copie Parquet écrite dans `data/parquet/creditcard/`.")

    text = "\n".join(report)
    (OUTPUT_DIR / "validation_report.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    spark.stop()
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
