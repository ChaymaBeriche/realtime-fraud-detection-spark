"""
Phase 2 — Data Understanding
Script 02 : analyse exploratoire (EDA) avec Spark SQL.

Principe Big Data : tous les calculs (comptages, moyennes, corrélations,
histogrammes) sont faits par Spark. Seuls les petits résultats agrégés
(quelques dizaines de lignes) sont ramenés en pandas pour les graphiques.

Entrée  : data/parquet/creditcard (produit par 01_load_and_validate.py)
Sorties : outputs/eda_summary.md
          outputs/figures/*.png

Usage (depuis la racine du projet) :
    python 2_data_understanding/02_eda.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")  # pas de fenêtre : on enregistre les images
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
from pyspark.sql.window import Window

import config as cfg

REPORTS_DIR = cfg.DATA_UNDERSTANDING_DIR / "outputs"
FIG_DIR = REPORTS_DIR / "figures"

# Palette : bleu = transactions normales, orange = fraudes
C_NORMAL, C_FRAUD = "#2a78d6", "#eb6834"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


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
    spark = cfg.spark_session("FraudDetection-EDA")
    df = spark.read.parquet(str(cfg.PARQUET_DIR))
    df = df.withColumn("hour", (F.floor(F.col("Time") / 3600) % 24).cast("int"))
    df.createOrReplaceTempView("tx")
    df.cache()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# Phase 2 — Analyse exploratoire (EDA)\n"]

    # ------------------------------------------------------------------ 1. classes
    classes = spark.sql("""
        SELECT Class, COUNT(*) AS n, ROUND(100 * COUNT(*) / SUM(COUNT(*)) OVER (), 3) AS pct
        FROM tx GROUP BY Class ORDER BY Class
    """).toPandas()
    n_normal, n_fraud = classes.loc[classes.Class == 0, "n"].item(), classes.loc[classes.Class == 1, "n"].item()
    lines += ["## 1. Répartition des classes\n",
              "| Classe | Transactions | Part |", "|---|---|---|",
              f"| Normal (0) | {n_normal:,} | {100 - classes.pct[1]:.3f} % |",
              f"| Fraude (1) | {n_fraud:,} | {classes.pct[1]:.3f} % |",
              f"\nRatio : **1 fraude pour {n_normal // n_fraud} transactions normales**. "
              "L'exactitude (accuracy) est donc inutilisable : un modèle qui prédit toujours "
              f"« normal » obtiendrait {100 - classes.pct[1]:.2f} %.\n"]

    # ------------------------------------------------------------------ 2. montants
    amounts = spark.sql("""
        SELECT Class,
               ROUND(AVG(Amount), 2)                         AS moyenne,
               ROUND(PERCENTILE_APPROX(Amount, 0.5), 2)      AS mediane,
               ROUND(PERCENTILE_APPROX(Amount, 0.95), 2)     AS p95,
               ROUND(MAX(Amount), 2)                         AS max,
               ROUND(100 * AVG(CAST(Amount = 0 AS INT)), 2)  AS pct_montant_zero
        FROM tx GROUP BY Class ORDER BY Class
    """).toPandas()
    lines += ["## 2. Montants (devise non précisée par la source)\n",
              "| Classe | Moyenne | Médiane | 95e centile | Max | Montant = 0 |", "|---|---|---|---|---|---|"]
    for _, r in amounts.iterrows():
        name = "Fraude" if r.Class == 1 else "Normal"
        lines.append(f"| {name} | {r.moyenne} | {r.mediane} | {r.p95} | {r['max']:,.2f} | {r.pct_montant_zero} % |")
    lines.append("\nLes fraudes ont une médiane plus faible mais une moyenne et un 95e centile "
                 "plus élevés : beaucoup de très petits montants (5,5 % valent 0) et une queue "
                 "de montants élevés. Le montant seul ne suffit pas à isoler la fraude.\n")

    # histogramme du log10(montant + 1), calculé par Spark (seaux de 0,1)
    hist = (df.withColumn("bin", F.floor(F.log10(F.col("Amount") + 1) * 10) / 10)
              .groupBy("Class", "bin").count()
              .withColumn("share", F.col("count") / F.sum("count").over(Window.partitionBy("Class")))
              .orderBy("Class", "bin").toPandas())
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for cls, color, label in [(0, C_NORMAL, "Normal"), (1, C_FRAUD, "Fraude")]:
        h = hist[hist.Class == cls]
        ax.step(h["bin"], 100 * h["share"], where="post", color=color, linewidth=2, label=label)
    style(ax, "Distribution des montants (part de chaque classe)", "log10(montant + 1)", "% des transactions")
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels(["0", "9", "99", "999", "9 999"])
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_montants.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------ 3. temps
    by_hour = spark.sql("""
        SELECT hour, COUNT(*) AS n, SUM(Class) AS frauds,
               100 * AVG(Class) AS fraud_rate
        FROM tx GROUP BY hour ORDER BY hour
    """).toPandas()
    peak = by_hour.loc[by_hour.fraud_rate.idxmax()]
    lines += ["## 3. Évolution selon l'heure\n",
              "`hour = floor(Time / 3600) mod 24` (heure relative au début de la collecte).\n",
              f"- Taux de fraude maximal à l'heure **{int(peak.hour)}** : {peak.fraud_rate:.2f} % "
              f"({int(peak.frauds)} fraudes sur {int(peak.n):,} transactions)",
              f"- Taux moyen : {100 * n_fraud / (n_normal + n_fraud):.3f} %",
              "- La nuit (peu de transactions), la part de fraude est bien plus élevée : "
              "l'heure est une variable utile.\n"]

    # deux panneaux superposés (pas de double axe)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    a1.bar(by_hour.hour, by_hour.n / 1000, color=C_NORMAL, width=0.8)
    style(a1, "Volume de transactions par heure", "", "milliers")
    a2.plot(by_hour.hour, by_hour.fraud_rate, color=C_FRAUD, linewidth=2, marker="o", markersize=4)
    style(a2, "Taux de fraude par heure", "heure relative", "% de fraudes")
    a2.annotate(f"{peak.fraud_rate:.2f} %", (peak.hour, peak.fraud_rate), xytext=(6, 0),
                textcoords="offset points", color=INK, va="center")
    a2.set_xticks(range(0, 24, 2))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_heures.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------ 4. variables
    feats = cfg.PCA_COLS + ["Amount", "hour"]
    corr_row = df.select([F.corr(c, "Class").alias(c) for c in feats]).first().asDict()
    means = df.groupBy("Class").agg(*[F.avg(c).alias(c) for c in feats]).toPandas().set_index("Class")
    corr = sorted(corr_row.items(), key=lambda kv: abs(kv[1]), reverse=True)

    lines += ["## 4. Variables les plus liées à la fraude\n",
              "Corrélation de Pearson avec `Class` et moyenne par classe (top 10).\n",
              "| Variable | Corrélation | Moyenne normal | Moyenne fraude |", "|---|---|---|---|"]
    for name, c in corr[:10]:
        lines.append(f"| {name} | {c:+.3f} | {means.loc[0, name]:+.2f} | {means.loc[1, name]:+.2f} |")
    weakest = ", ".join(n for n, _ in corr[-5:])
    lines.append(f"\nVariables les moins liées : {weakest}. Les corrélations restent modestes "
                 "car la classe fraude est minuscule ; les écarts de moyenne, eux, sont nets "
                 "(V14, V17, V12…). Un modèle non linéaire devrait en tirer parti.\n")

    top = corr[:15][::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh([n for n, _ in top], [c for _, c in top],
            color=[C_FRAUD if c > 0 else C_NORMAL for _, c in top], height=0.7)
    ax.axvline(0, color=INK_2, linewidth=0.8)
    vals = [c for _, c in top]
    ax.set_xlim(min(vals) - 0.06, max(vals) + 0.06)
    style(ax, "Corrélation avec la fraude (15 variables les plus liées)", "corrélation de Pearson", "")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    for i, (_, c) in enumerate(top):
        ax.text(c + (0.01 if c > 0 else -0.01), i, f"{c:+.2f}", va="center",
                ha="left" if c > 0 else "right", color=INK_2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_correlations.png", dpi=150)
    plt.close(fig)

    # distributions des 4 variables les plus discriminantes
    top4 = [n for n, _ in corr if n not in ("Amount", "hour")][:4]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for ax, name in zip(axes.flat, top4):
        lo, hi = df.approxQuantile(name, [0.001, 0.999], 0.001)
        step = (hi - lo) / 40
        h = (df.filter(F.col(name).between(lo, hi))
               .withColumn("bin", F.floor((F.col(name) - lo) / step) * step + lo)
               .groupBy("Class", "bin").count().toPandas())
        for cls, color, label in [(0, C_NORMAL, "Normal"), (1, C_FRAUD, "Fraude")]:
            s = h[h.Class == cls].sort_values("bin")
            ax.step(s["bin"], 100 * s["count"] / s["count"].sum(), where="post",
                    color=color, linewidth=2, label=label)
        style(ax, name, "", "%")
    axes.flat[0].legend(frameon=False)
    fig.suptitle("Distribution des 4 variables les plus discriminantes", x=0.02, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_top_variables.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------ 5. doublons
    dups = spark.sql("""
        SELECT Class, SUM(cnt - 1) AS extra_rows
        FROM (SELECT *, COUNT(*) AS cnt FROM tx GROUP BY ALL)
        WHERE cnt > 1 GROUP BY Class ORDER BY Class
    """).toPandas()
    d = {int(r.Class): int(r.extra_rows) for _, r in dups.iterrows()}
    lines += ["## 5. Doublons\n",
              f"- Lignes en double (copies en trop) : {sum(d.values()):,}",
              f"  - dont normales : {d.get(0, 0):,}",
              f"  - dont fraudes : {d.get(1, 0):,}",
              "- Décision : supprimer les doublons exacts avant le split train/test "
              "(phase 3 — Data Preparation), sinon une même transaction pourrait se retrouver des deux côtés.\n"]

    lines += ["## Graphiques\n",
              "- `figures/01_montants.png` — distribution des montants",
              "- `figures/02_heures.png` — volume et taux de fraude par heure",
              "- `figures/03_correlations.png` — corrélations avec la fraude",
              "- `figures/04_top_variables.png` — distributions des variables clés\n"]

    (REPORTS_DIR / "eda_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    spark.stop()


if __name__ == "__main__":
    main()
