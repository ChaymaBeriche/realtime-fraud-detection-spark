"""
Configuration partagée par toutes les phases du projet (CRISP-DM).

Chaque script d'une phase l'importe ainsi :
    import sys; from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import config as cfg
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- Données -----------------------------------------------------------------
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "archive" / "creditcard.csv"      # fichier Kaggle d'origine (jamais modifié)
PARQUET_DIR = DATA_DIR / "parquet" / "creditcard"      # copie typée produite en phase 2

# --- Dossiers des phases CRISP-DM ----------------------------------------------
BUSINESS_DIR = ROOT / "1_business_understanding"
DATA_UNDERSTANDING_DIR = ROOT / "2_data_understanding"
DATA_PREPARATION_DIR = ROOT / "3_data_preparation"
MODELING_DIR = ROOT / "4_modeling"
EVALUATION_DIR = ROOT / "5_evaluation"
DEPLOYMENT_DIR = ROOT / "6_deployment"

# --- Valeurs officielles du dataset (contrôle d'intégrité) ---------------------
EXPECTED_ROWS = 284_807
EXPECTED_FRAUDS = 492

# --- Colonnes ------------------------------------------------------------------
LABEL_COL = "Class"                                    # 1 = fraude, 0 = normal
PCA_COLS = [f"V{i}" for i in range(1, 29)]
FEATURE_COLS = ["Time", "Amount"] + PCA_COLS

RANDOM_SEED = 42

# Nombre fixe de threads Spark. Avec "local[*]", Spark utiliserait tous les cœurs de
# la machine : le découpage des données en partitions changerait d'un ordinateur à
# l'autre, et avec lui les tirages aléatoires (Random Forest, GBT, KMeans).
# Une valeur fixe rend les résultats identiques sur toutes les machines.
SPARK_THREADS = 4

# --- Phase 3 : préparation -----------------------------------------------------
PREPARED_DIR = DATA_DIR / "prepared"                   # train / validation / test (Parquet)
SPLITS = ["train", "validation", "test"]
# Découpage temporel, en part de la durée totale (~48 h) :
# train = [0 ; 70 %[, validation = [70 % ; 85 %[, test = [85 % ; 100 %]
TRAIN_END_FRACTION = 0.70
VALIDATION_END_FRACTION = 0.85

# Variables d'entrée des modèles (phase 4). Time n'est pas utilisé directement :
# on garde "hour" (heure relative) et "log_amount" (montant en échelle log).
MODEL_FEATURES = PCA_COLS + ["log_amount", "hour"]

# --- Phase 4 : modélisation ----------------------------------------------------
MODELS_DIR = ROOT / "models"                           # pipelines MLlib sauvegardés
WEIGHT_COL = "weight"                                  # poids de classe (déséquilibre)

# --- Phase 5 : évaluation ------------------------------------------------------
DECISION_FILE = MODELS_DIR / "decision.json"           # modèle retenu + seuil (utilisé en phase 6)
F_BETA = 2.0                                           # F2 : une fraude manquée compte 2x plus qu'une fausse alerte
BOOTSTRAP_SAMPLES = 1000                               # rééchantillonnages pour les intervalles de confiance

# --- Phase 6 : déploiement (streaming) -----------------------------------------
STREAM_DIR = DATA_DIR / "stream"
STREAM_INPUT_DIR = STREAM_DIR / "input"                # dossier surveillé par Spark (fichiers du flux)
STREAM_OUTPUT_DIR = STREAM_DIR / "output"              # résultats du streaming
STREAM_CHECKPOINT_DIR = STREAM_DIR / "checkpoints"     # état des requêtes (reprise après panne)
STREAM_LABELS_FILE = STREAM_DIR / "labels.csv"         # vraies classes, gardées À PART (contrôle uniquement)
STREAM_DONE_FLAG = STREAM_INPUT_DIR / "_DONE"          # signal de fin du simulateur ("_" : ignoré par Spark)
STREAM_READY_FLAG = STREAM_DIR / "_READY"              # signal « détection prête » (le simulateur l'attend)
STREAM_SPEED = 60                                      # x60 : 1 seconde réelle = 1 minute de transactions
STREAM_TRIGGER = "2 seconds"                           # Spark traite les nouveaux fichiers toutes les 2 s
# Date de départ conventionnelle pour transformer Time (secondes) en horodatage.
# La source indique seulement « septembre 2013 » : le jour exact n'est pas connu.
STREAM_BASE_TIME = "2013-09-01 00:00:00"


def spark_session(app_name: str = "FraudDetection"):
    """Crée une session Spark locale avec des logs réduits."""
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.appName(app_name)
        .master(f"local[{SPARK_THREADS}]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark
