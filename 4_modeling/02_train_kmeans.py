"""
Phase 4 — Modeling
Script 02 : détection d'anomalies non supervisée avec KMeans.

Principe :
  - KMeans regroupe les transactions du train en k groupes SANS regarder l'étiquette
  - k est choisi parmi {5, 10, 20} avec le score silhouette (qualité des groupes),
    donc lui aussi sans utiliser les étiquettes
  - deux scores d'anomalie sont comparés :
      A. distance au centre de SON groupe (version simple)
      B. distance au centre du GRAND groupe le plus proche (méthode CBLOF) :
         un « grand » groupe contient au moins 1 % du train. Une transaction qui
         tombe dans un petit groupe rare est loin de tous les grands groupes.
    Le score B a été ajouté après avoir constaté que KMeans range la plupart des
    fraudes dans de petits groupes à part : elles sont alors proches du centre de
    leur groupe, et le score A ne les voit pas.

Les étiquettes ne servent qu'à MESURER le résultat (validation) et à décrire
les groupes obtenus ; elles n'interviennent jamais dans l'apprentissage.

Entrées : data/prepared/train, data/prepared/validation      (phase 3)
          models/<meilleur modèle supervisé>, outputs/best_models.json (script 01)
Sorties : models/kmeans/                 pipeline KMeans
          models/kmeans_meta.json        grands groupes retenus (pour le score B)
          4_modeling/outputs/kmeans_report.md
          4_modeling/outputs/figures/kmeans_scores.png

Usage (depuis la racine du projet) :
    python 4_modeling/02_train_kmeans.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import BinaryClassificationEvaluator, ClusteringEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

import config as cfg

OUTPUT_DIR = cfg.MODELING_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
K_VALUES = [5, 10, 20]
LARGE_CLUSTER_SHARE = 0.01        # « grand » groupe : au moins 1 % des transactions du train
ALERT_SHARES = [0.005, 0.02]      # budgets d'alertes comparés : 0,5 % et 2 % des transactions

C_NORMAL, C_FRAUD = "#2a78d6", "#eb6834"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def squared_distance(vec, center):
    """Distance au carré entre la colonne vecteur et un centre, en Spark SQL."""
    sq = F.zip_with(vec, F.array(*[F.lit(float(x)) for x in center]), lambda a, b: (a - b) * (a - b))
    return F.aggregate(sq, F.lit(0.0), lambda acc, x: acc + x)


def add_scores(df, centers, large_ids):
    """Score A : distance au centre de son propre groupe.
       Score B : distance au centre du grand groupe le plus proche (CBLOF)."""
    vec = vector_to_array("features")
    all_centers = F.array(*[F.array(*[F.lit(float(x)) for x in c]) for c in centers])
    own = F.element_at(all_centers, F.col("cluster") + 1)
    own_sq = F.aggregate(F.zip_with(vec, own, lambda a, b: (a - b) * (a - b)),
                         F.lit(0.0), lambda acc, x: acc + x)
    large_sq = [squared_distance(vec, centers[i]) for i in large_ids]
    nearest_large = F.least(*large_sq) if len(large_sq) > 1 else large_sq[0]
    return (df.withColumn("score_center", F.sqrt(own_sq))
              .withColumn("score_cblof", F.sqrt(nearest_large)))


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2)


def main():
    spark = cfg.spark_session("FraudDetection-KMeans")
    train = spark.read.parquet(str(cfg.PREPARED_DIR / "train")).cache()
    valid = spark.read.parquet(str(cfg.PREPARED_DIR / "validation")).cache()
    n_train = train.count()
    n_valid = valid.count()
    n_valid_fraud = valid.filter(F.col(cfg.LABEL_COL) == 1).count()

    assembler = VectorAssembler(inputCols=cfg.MODEL_FEATURES, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features",
                            withMean=True, withStd=True)
    silhouette = ClusteringEvaluator(featuresCol="features", predictionCol="cluster")

    def auc(df, col, metric):
        return BinaryClassificationEvaluator(labelCol=cfg.LABEL_COL, rawPredictionCol=col,
                                             metricName=metric).evaluate(df)

    # ------------------------------------------------------------ 1. choix de k (sans étiquettes)
    runs = []
    for k in K_VALUES:
        km = KMeans(k=k, featuresCol="features", predictionCol="cluster",
                    seed=cfg.RANDOM_SEED, maxIter=50)
        model = Pipeline(stages=[assembler, scaler, km]).fit(train)
        sil = silhouette.evaluate(model.transform(train))
        runs.append(dict(k=k, silhouette=sil, model=model))
        print(f"k={k:<3} silhouette={sil:.4f}")
    best = max(runs, key=lambda r: r["silhouette"])
    model, k = best["model"], best["k"]
    centers = model.stages[-1].clusterCenters()

    # ------------------------------------------------------------ 2. grands groupes (sans étiquettes)
    train_c = model.transform(train)
    groups = (train_c.groupBy("cluster")
                     .agg(F.count("*").alias("n"), F.sum(cfg.LABEL_COL).alias("frauds"))
                     .orderBy("n").toPandas())
    large_ids = sorted(int(c) for c in groups.loc[groups.n >= LARGE_CLUSTER_SHARE * n_train, "cluster"])

    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.write().overwrite().save(str(cfg.MODELS_DIR / "kmeans"))
    (cfg.MODELS_DIR / "kmeans_meta.json").write_text(json.dumps(
        {"k": k, "large_cluster_share": LARGE_CLUSTER_SHARE, "large_cluster_ids": large_ids},
        indent=2), encoding="utf-8")

    # ------------------------------------------------------------ 3. scores sur la validation
    # le meilleur modèle supervisé (script 01) est appliqué à la même table
    best_models = json.loads((OUTPUT_DIR / "best_models.json").read_text(encoding="utf-8"))
    sup_family = max(best_models, key=lambda f: best_models[f]["auc_pr"])
    sup_label = {"logistic_regression": "Régression logistique", "random_forest": "Random Forest",
                 "gbt": "GBT"}[sup_family]
    sup_model = PipelineModel.load(str(cfg.ROOT / best_models[sup_family]["path"]))
    with_sup = sup_model.transform(valid).select(
        *valid.columns, vector_to_array("probability")[1].alias("sup_score"))
    scored = (add_scores(model.transform(with_sup), centers, large_ids)
              .select(cfg.LABEL_COL, "Amount", "sup_score", "score_center", "score_cblof").cache())

    metrics = {col: (auc(scored, col, "areaUnderPR"), auc(scored, col, "areaUnderROC"))
               for col in ("score_center", "score_cblof")}

    # ------------------------------------------------------------ 4. budgets d'alertes
    def frauds_in_top(col, n_alerts):
        cut = scored.orderBy(F.desc(col)).limit(n_alerts).agg(F.min(col)).first()[0]
        return F.col(col) >= cut

    budget_rows = []
    frauds = scored.filter(F.col(cfg.LABEL_COL) == 1)
    for share in ALERT_SHARES:
        n_alerts = int(round(share * n_valid))
        km_a, sup_a = frauds_in_top("score_cblof", n_alerts), frauds_in_top("sup_score", n_alerts)
        km_c = frauds_in_top("score_center", n_alerts)
        r = frauds.agg(
            F.sum(km_c.cast("int")).alias("center"),
            F.sum(km_a.cast("int")).alias("cblof"),
            F.sum(sup_a.cast("int")).alias("sup"),
            F.sum((km_a & ~sup_a).cast("int")).alias("cblof_only"),
            F.sum((km_a | sup_a).cast("int")).alias("either"),
        ).first().asDict()
        r.update(share=share, n_alerts=n_alerts)
        budget_rows.append(r)

    # description des alertes du score B au plus petit budget : fraudes ? montants ?
    n_top = budget_rows[0]["n_alerts"]
    top_b = scored.orderBy(F.desc("score_cblof")).limit(n_top)
    top_desc = top_b.agg(F.sum(cfg.LABEL_COL).alias("frauds"),
                         F.expr("percentile_approx(Amount, 0.5)").alias("median_amount")).first()
    median_amount_all = scored.agg(F.expr("percentile_approx(Amount, 0.5)")).first()[0]

    # ------------------------------------------------------------ 5. graphique
    pdf = scored.select(F.log10("score_cblof").alias("s"), cfg.LABEL_COL).toPandas()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    lo, hi = pdf["s"].min(), pdf["s"].max()
    for cls, color, label in [(0, C_NORMAL, "Normal"), (1, C_FRAUD, "Fraude")]:
        ax.hist(pdf.loc[pdf[cfg.LABEL_COL] == cls, "s"], bins=50, range=(lo, hi), density=True,
                histtype="step", linewidth=2, color=color, label=label)
    style(ax, f"Score d'anomalie KMeans (k={k}, score B) — validation",
          "log10(distance au grand groupe le plus proche)", "densité")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "kmeans_scores.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------ 6. rapport
    n_train_fraud = int(groups.frauds.sum())
    small = groups[~groups.cluster.isin(large_ids)]
    lines = [
        "# Phase 4 — Détection d'anomalies avec KMeans\n",
        "KMeans est appris sur le train **sans les étiquettes**. Les étiquettes servent "
        "uniquement à mesurer le résultat et à décrire les groupes obtenus.\n",
        "## 1. Choix de k (silhouette)\n",
        "| k | Silhouette |", "|---|---|",
    ]
    for r in runs:
        lines.append(f"| {r['k']}{' ← retenu' if r is best else ''} | {r['silhouette']:.4f} |")

    lines += [
        "", f"## 2. Groupes obtenus (k = {k})\n",
        f"{len(large_ids)} grands groupes (≥ {LARGE_CLUSTER_SHARE:.0%} du train) et "
        f"{k - len(large_ids)} petits groupes. Composition des petits groupes "
        "(étiquettes utilisées ici pour décrire, pas pour apprendre) :\n",
        "| Groupe | Transactions | dont fraudes |", "|---|---|---|",
    ]
    for g in small.itertuples():
        lines.append(f"| {int(g.cluster)} | {int(g.n):,} | {int(g.frauds)} |")
    # --- phrases calculées à partir des résultats (aucune conclusion écrite à l'avance)
    def compare(a, b, tol=0.10):
        """Compare deux valeurs : 'meilleur', 'comparable' ou 'moins bon' (écart relatif)."""
        if a > b * (1 + tol):
            return "meilleur"
        if a < b * (1 - tol):
            return "moins bon"
        return "comparable"

    small_frauds, small_n = int(small.frauds.sum()), int(small.n.sum())
    fraud_share = small_frauds / n_train_fraud if n_train_fraud else 0.0
    tx_share = small_n / n_train
    if fraud_share >= 0.5:
        structure = (f"KMeans range **{fraud_share:.0%} des fraudes** dans des petits groupes "
                     f"qui ne représentent que {tx_share:.1%} des transactions, sans avoir vu les "
                     "étiquettes : les fraudes forment des groupes à part.")
    else:
        structure = (f"Seulement {fraud_share:.0%} des fraudes tombent dans ces petits groupes : "
                     "KMeans ne sépare pas clairement les fraudes des transactions normales.")

    auc_a, auc_b = metrics["score_center"][0], metrics["score_cblof"][0]
    verdict_b = compare(auc_b, auc_a)
    scores_text = {
        "meilleur": "Le score B est **meilleur** que le score A sur cette validation.",
        "comparable": "Les scores A et B sont **comparables** sur cette validation "
                      "(moins de 10 % d'écart d'AUC-PR).",
        "moins bon": "Le score B est **moins bon** que le score A sur cette validation.",
    }[verdict_b]

    sil_max = max(r["silhouette"] for r in runs)
    sil_text = ("Silhouette faible (< 0,25) pour toutes les valeurs de k : les transactions ne "
                "forment pas de groupes nettement séparés, et le choix de k est fragile."
                if sil_max < 0.25 else
                "Silhouette correcte : les groupes sont assez bien séparés.")

    conclusion = []
    conclusion.append(f"- {structure}")
    conclusion.append(f"- {sil_text}")
    top_precision = top_desc["frauds"] / n_top
    conclusion.append(
        f"- Parmi les {n_top} transactions les plus inhabituelles selon le score B, "
        f"{int(top_desc['frauds'])} fraude(s) seulement ({top_precision:.1%}). Leur montant médian "
        f"est de {top_desc['median_amount']:.2f}, contre {median_amount_all:.2f} pour l'ensemble "
        "de la validation." + (" Inhabituel ne veut donc pas dire frauduleux."
                               if top_precision < 0.5 else ""))
    for r in budget_rows:
        v = compare(r["cblof"], r["sup"])
        word = {"meilleur": "trouve plus de fraudes que", "comparable": "fait jeu égal avec",
                "moins bon": "trouve moins de fraudes que"}[v]
        conclusion.append(
            f"- Avec {r['n_alerts']} alertes, KMeans (score B) {word} le {sup_label} "
            f"({r['cblof']} contre {r['sup']}) et trouve {r['cblof_only']} fraude(s) que le "
            f"{sup_label} manque.")
    if all(r["cblof_only"] == 0 for r in budget_rows):
        conclusion.append(f"- KMeans n'apporte donc **pas de complément mesurable** au {sup_label} "
                          "sur ces données.")
    else:
        conclusion.append(f"- KMeans apporte un **complément** : certaines fraudes ne sont vues que par lui.")

    lines += [
        "",
        f"Les petits groupes contiennent {small_frauds} des {n_train_fraud} fraudes du train, "
        f"pour {small_n:,} transactions sur {n_train:,}.\n",
        "## 3. Deux scores d'anomalie (validation)\n",
        "| Score | Définition | AUC-PR | AUC-ROC |", "|---|---|---|---|",
        f"| A | distance au centre de son groupe | {auc_a:.4f} | {metrics['score_center'][1]:.4f} |",
        f"| B (CBLOF) | distance au grand groupe le plus proche | {auc_b:.4f} | "
        f"{metrics['score_cblof'][1]:.4f} |",
        "",
        "Idée du score B : si des fraudes forment leur propre petit groupe, elles sont proches "
        "du centre de ce groupe et le score A les manque ; le score B mesure plutôt leur "
        "distance aux comportements les plus courants (grands groupes).",
        "",
        scores_text + "\n",
        f"## 4. Comparaison avec le meilleur modèle supervisé ({sup_label})\n",
        f"Même nombre d'alertes pour chaque méthode. Validation : {n_valid:,} transactions, "
        f"{n_valid_fraud} fraudes.\n",
        f"| Budget d'alertes | Score A | Score B | {sup_label} | B seulement | B ou {sup_label} |",
        "|---|---|---|---|---|---|",
    ]
    for r in budget_rows:
        lines.append(f"| {r['share']:.1%} ({r['n_alerts']} alertes) | {r['center']} | {r['cblof']} | "
                     f"{r['sup']} | {r['cblof_only']} | {r['either']} |")
    lines += [
        "",
        "Chaque case donne le nombre de fraudes trouvées parmi les alertes.\n",
        "## 5. Conclusion (calculée à partir des résultats ci-dessus)",
        *conclusion,
        "",
        "## Graphique",
        "- `figures/kmeans_scores.png` — distribution du score B (normal / fraude)",
    ]
    text = "\n".join(lines)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "kmeans_report.md").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
    spark.stop()


if __name__ == "__main__":
    main()
