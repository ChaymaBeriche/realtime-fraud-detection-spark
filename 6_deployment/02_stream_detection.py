"""
Phase 6 — Deployment
Script 02 : détection de fraude en temps réel avec Spark Structured Streaming.

Fonctionnement :
  1. Spark surveille le dossier data/stream/input/ : chaque nouveau fichier déposé par
     le simulateur (script 01) devient un petit lot (« micro-batch ») traité en continu
  2. pour chaque transaction reçue :
       - mêmes variables qu'en phase 3 : hour = floor(Time/3600) mod 24, log_amount = log(1+Amount)
       - application du pipeline MLlib retenu en phase 5 (Random Forest) -> probabilité de fraude
       - alerte si probabilité >= seuil choisi en phase 5 (models/decision.json)
  3. deux requêtes de streaming tournent en parallèle :
       A. « détection » : chaque micro-batch est écrit dans data/stream/output/scored/
          (toutes les transactions notées) et les alertes sont affichées à l'écran
       B. « fenêtres » : agrégation par minute de temps de transaction (fenêtres de
          1 minute, watermark de 2 minutes) -> data/stream/output/windows/
  4. l'état des requêtes est sauvegardé (checkpoints) : après une panne, Spark reprend
     là où il s'était arrêté sans perdre ni dupliquer de transactions

La vraie classe n'arrive jamais dans le flux : la détection ne s'appuie que sur le modèle.

Le script s'arrête tout seul quand le simulateur a fini (fichier _DONE) et que tous
les fichiers ont été traités.

Usage (depuis la racine du projet) — lancer ce script, puis le simulateur :
    python 6_deployment/02_stream_detection.py
"""
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, LongType, StructField, StructType

import config as cfg

# Schéma des fichiers envoyés par le simulateur (pas de colonne Class)
SCHEMA = StructType(
    [StructField("transaction_id", LongType(), False),
     StructField("Time", DoubleType(), False),
     StructField("Amount", DoubleType(), False)]
    + [StructField(c, DoubleType(), False) for c in cfg.PCA_COLS]
    + [StructField("sent_at", DoubleType(), False)]
)

SCORED_DIR = cfg.STREAM_OUTPUT_DIR / "scored"
WINDOWS_DIR = cfg.STREAM_OUTPUT_DIR / "windows"
BATCHES_FILE = cfg.STREAM_OUTPUT_DIR / "batches.csv"


def reset_stream_folders():
    """Nouvelle démonstration : on repart de zéro (sorties, checkpoints, ancien flux)."""
    for d in (cfg.STREAM_OUTPUT_DIR, cfg.STREAM_CHECKPOINT_DIR):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    cfg.STREAM_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in cfg.STREAM_INPUT_DIR.iterdir():
        f.unlink()
    BATCHES_FILE.write_text("batch_id,rows,alerts,processed_at,latency_median_s,latency_max_s\n",
                            encoding="utf-8")


def main():
    decision = json.loads(cfg.DECISION_FILE.read_text(encoding="utf-8"))
    threshold = float(decision["threshold"])

    reset_stream_folders()
    spark = cfg.spark_session("FraudDetection-Streaming")
    model = PipelineModel.load(str(cfg.ROOT / decision["model_path"]))

    # ------------------------------------------------------------ source : dossier surveillé
    raw = (spark.readStream
                .schema(SCHEMA)
                .option("header", True)
                .option("maxFilesPerTrigger", 30)        # au plus 30 fichiers par micro-batch
                .csv(str(cfg.STREAM_INPUT_DIR)))

    # ------------------------------------------------------------ mêmes variables qu'en phase 3
    features = (raw.withColumn("hour", (F.floor(F.col("Time") / 3600) % 24).cast("int"))
                   .withColumn("log_amount", F.log1p(F.col("Amount")))
                   .withColumn("event_time",
                               F.timestamp_seconds(F.unix_timestamp(F.lit(cfg.STREAM_BASE_TIME))
                                                   + F.col("Time"))))

    # ------------------------------------------------------------ modèle + seuil
    scored = (model.transform(features)
                   .withColumn("score", vector_to_array("probability")[1])
                   .withColumn("alert", F.col("score") >= F.lit(threshold))
                   .select("transaction_id", "event_time", "Time", "Amount", "hour",
                           "score", "alert", "sent_at"))

    # ------------------------------------------------------------ requête A : détection
    def handle_batch(batch_df, batch_id):
        processed_at = time.time()
        batch_df = batch_df.withColumn("processed_at", F.lit(processed_at)).cache()
        batch_df.write.mode("append").parquet(str(SCORED_DIR))
        stats = batch_df.agg(
            F.count("*").alias("rows"),
            F.sum(F.col("alert").cast("int")).alias("alerts"),
            F.expr(f"percentile_approx({processed_at} - sent_at, 0.5)").alias("lat_med"),
            F.max(F.lit(processed_at) - F.col("sent_at")).alias("lat_max"),
        ).first()
        if stats["rows"]:
            with BATCHES_FILE.open("a", encoding="utf-8") as fh:
                fh.write(f"{batch_id},{stats['rows']},{stats['alerts']},{processed_at:.3f},"
                         f"{stats['lat_med']:.3f},{stats['lat_max']:.3f}\n")
            print(f"[lot {batch_id:>3}] {stats['rows']:>5} transactions, {stats['alerts']} alerte(s), "
                  f"latence médiane {stats['lat_med']:.1f} s")
            for a in (batch_df.filter("alert").orderBy("event_time")
                              .select("transaction_id", "event_time", "Amount", "score").collect()):
                print(f"   ALERTE  transaction {a['transaction_id']:>6}  "
                      f"{a['event_time']:%H:%M:%S}  montant {a['Amount']:>9.2f}  "
                      f"probabilité {a['score']:.3f}")
        batch_df.unpersist()

    detection = (scored.writeStream
                       .queryName("detection")
                       .foreachBatch(handle_batch)
                       .option("checkpointLocation", str(cfg.STREAM_CHECKPOINT_DIR / "detection"))
                       .trigger(processingTime=cfg.STREAM_TRIGGER)
                       .start())

    # ------------------------------------------------------------ requête B : fenêtres d'1 minute
    windows = (scored.withWatermark("event_time", "2 minutes")
                     .groupBy(F.window("event_time", "1 minute"))
                     .agg(F.count("*").alias("transactions"),
                          F.sum(F.col("alert").cast("int")).alias("alerts"),
                          F.round(F.sum("Amount"), 2).alias("amount"))
                     .select(F.col("window.start").alias("window_start"),
                             F.col("window.end").alias("window_end"),
                             "transactions", "alerts", "amount"))
    windows_query = (windows.writeStream
                            .queryName("windows")
                            .outputMode("append")
                            .format("parquet")
                            .option("path", str(WINDOWS_DIR))
                            .option("checkpointLocation", str(cfg.STREAM_CHECKPOINT_DIR / "windows"))
                            .trigger(processingTime=cfg.STREAM_TRIGGER)
                            .start())

    # ------------------------------------------------------------ prêt : le simulateur peut envoyer
    cfg.STREAM_READY_FLAG.write_text(str(time.time()), encoding="utf-8")
    print(f"\nDétection prête : modèle {decision['model']}, seuil {threshold:.4f}.")
    print(f"Surveillance de {cfg.STREAM_INPUT_DIR}")
    print("Lancez maintenant le simulateur (6_deployment/01_stream_simulator.py) "
          "dans un autre terminal.\nInterface Spark : http://localhost:4040\n")

    try:
        while not cfg.STREAM_DONE_FLAG.exists():
            if detection.exception() or windows_query.exception():
                raise RuntimeError(detection.exception() or windows_query.exception())
            time.sleep(2)
        # le simulateur a fini : on traite les derniers fichiers puis on s'arrête
        detection.processAllAvailable()
        windows_query.processAllAvailable()
    finally:
        detection.stop()
        windows_query.stop()
        cfg.STREAM_READY_FLAG.unlink(missing_ok=True)
        spark.stop()
    print("\nFlux terminé. Lancez le rapport : python 6_deployment/03_stream_report.py")


if __name__ == "__main__":
    main()
