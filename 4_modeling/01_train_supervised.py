"""
Phase 4 — Modeling
Script 01 : modèles supervisés MLlib (régression logistique, Random Forest, GBT).

Pour chaque modèle :
  - pipeline MLlib : VectorAssembler -> StandardScaler -> classifieur
    (la standardisation est apprise sur le train uniquement)
  - deux variantes : sans poids, et avec poids de classe (déséquilibre)
  - une petite grille de paramètres, chaque réglage étant évalué sur la VALIDATION
    (pas de CrossValidator : il mélangerait les lignes et casserait l'ordre du temps)

Métrique de comparaison : AUC-PR (aire sous la courbe précision-rappel) sur la
validation. L'AUC-ROC est donnée à titre indicatif.

Le meilleur réglage de chaque famille de modèle est sauvegardé dans models/.
Le choix du seuil de décision et le score final sur le test se font en phase 5.

Entrées : data/prepared/train, data/prepared/validation      (phase 3)
Sorties : models/{logistic_regression,random_forest,gbt}/
          4_modeling/outputs/supervised_results.csv
          4_modeling/outputs/model_comparison.md
          4_modeling/outputs/best_models.json
          4_modeling/outputs/figures/pr_curves_validation.png
          4_modeling/outputs/figures/feature_importance.png

Usage (depuis la racine du projet) :
    python 4_modeling/01_train_supervised.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.classification import (GBTClassifier, LogisticRegression,
                                       RandomForestClassifier)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

import config as cfg

OUTPUT_DIR = cfg.MODELING_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"

# Couleurs : une par famille de modèle (bleu, orange, vert d'eau)
COLORS = {"logistic_regression": "#2a78d6", "random_forest": "#eb6834", "gbt": "#1baf7a"}
LABELS = {"logistic_regression": "Régression logistique", "random_forest": "Random Forest",
          "gbt": "GBT"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def build_grid():
    """Liste de (famille, nom du réglage, fonction qui crée le classifieur)."""
    common = dict(featuresCol="features", labelCol=cfg.LABEL_COL)
    grid = []
    for reg in [0.0, 0.01, 0.1]:
        grid.append(("logistic_regression", f"regParam={reg}",
                     lambda w, reg=reg: LogisticRegression(
                         regParam=reg, maxIter=100, **common, **w)))
    for depth in [5, 10]:
        grid.append(("random_forest", f"numTrees=100, maxDepth={depth}",
                     lambda w, depth=depth: RandomForestClassifier(
                         numTrees=100, maxDepth=depth, seed=cfg.RANDOM_SEED, **common, **w)))
    for iters in [50, 100]:
        for depth in [3, 5]:
            grid.append(("gbt", f"maxIter={iters}, maxDepth={depth}",
                         lambda w, iters=iters, depth=depth: GBTClassifier(
                             maxIter=iters, maxDepth=depth, stepSize=0.1,
                             seed=cfg.RANDOM_SEED, **common, **w)))
    return grid


def add_class_weights(df, n_total, n_fraud):
    """Poids « équilibrés » : chaque classe pèse autant au total.
    w(classe) = N / (2 * n_classe)  ->  ~0,5 pour normal, ~250 pour fraude."""
    w_fraud = n_total / (2.0 * n_fraud)
    w_normal = n_total / (2.0 * (n_total - n_fraud))
    df = df.withColumn(cfg.WEIGHT_COL,
                       F.when(F.col(cfg.LABEL_COL) == 1, w_fraud).otherwise(w_normal))
    return df, w_normal, w_fraud


def pr_curve(scores, labels):
    """Courbe précision-rappel à partir de scores (calculée sur un petit tableau)."""
    order = np.argsort(-scores)
    y = labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / (tp + fp)
    recall = tp / y.sum()
    return recall, precision


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2)


def main():
    spark = cfg.spark_session("FraudDetection-Supervised")
    train = spark.read.parquet(str(cfg.PREPARED_DIR / "train"))
    valid = spark.read.parquet(str(cfg.PREPARED_DIR / "validation"))

    n_train = train.count()
    n_train_fraud = train.filter(F.col(cfg.LABEL_COL) == 1).count()
    n_valid_fraud = valid.filter(F.col(cfg.LABEL_COL) == 1).count()
    train, w_normal, w_fraud = add_class_weights(train, n_train, n_train_fraud)
    train.cache()
    valid.cache()

    assembler = VectorAssembler(inputCols=cfg.MODEL_FEATURES, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features",
                            withMean=True, withStd=True)
    ev_pr = BinaryClassificationEvaluator(labelCol=cfg.LABEL_COL, rawPredictionCol="probability",
                                          metricName="areaUnderPR")
    ev_roc = BinaryClassificationEvaluator(labelCol=cfg.LABEL_COL, rawPredictionCol="probability",
                                           metricName="areaUnderROC")

    # ------------------------------------------------------------ entraînement
    results, best = [], {}
    grid = build_grid()
    for i, (family, setting, make) in enumerate(grid, 1):
        for weighted in (False, True):
            w = {"weightCol": cfg.WEIGHT_COL} if weighted else {}
            pipeline = Pipeline(stages=[assembler, scaler, make(w)])
            t0 = time.time()
            model = pipeline.fit(train)
            pred = model.transform(valid)
            auc_pr, auc_roc = ev_pr.evaluate(pred), ev_roc.evaluate(pred)
            seconds = time.time() - t0
            row = dict(family=family, setting=setting,
                       weights="avec poids" if weighted else "sans poids",
                       auc_pr=auc_pr, auc_roc=auc_roc, seconds=seconds)
            results.append(row)
            print(f"[{i}/{len(grid)}] {LABELS[family]:<22} {setting:<28} "
                  f"{row['weights']:<10}  AUC-PR={auc_pr:.4f}  AUC-ROC={auc_roc:.4f}  ({seconds:.0f} s)")
            if family not in best or auc_pr > best[family]["row"]["auc_pr"]:
                best[family] = {"row": row, "model": model}

    # ------------------------------------------------------------ sauvegarde des modèles
    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_json = {}
    for family, b in best.items():
        path = cfg.MODELS_DIR / family
        b["model"].write().overwrite().save(str(path))
        best_json[family] = {k: b["row"][k] for k in ("setting", "weights", "auc_pr", "auc_roc")}
        best_json[family]["path"] = f"models/{family}"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(results)
    res.to_csv(OUTPUT_DIR / "supervised_results.csv", index=False, float_format="%.5f")
    (OUTPUT_DIR / "best_models.json").write_text(json.dumps(best_json, indent=2, ensure_ascii=False),
                                                 encoding="utf-8")

    # ------------------------------------------------------------ courbes PR (validation)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    base_rate = n_valid_fraud / valid.count()
    for family, b in best.items():
        sc = (b["model"].transform(valid)
              .select(vector_to_array("probability")[1].alias("score"), cfg.LABEL_COL)
              .toPandas())
        rec, prec = pr_curve(sc["score"].to_numpy(), sc[cfg.LABEL_COL].to_numpy())
        ax.plot(rec, prec, color=COLORS[family], linewidth=2,
                label=f"{LABELS[family]} (AUC-PR {b['row']['auc_pr']:.3f})")
    # un modèle au hasard aurait une précision égale au taux de fraude (~0,1 %), à peine visible
    style(ax, f"Courbes précision-rappel — validation ({n_valid_fraud} fraudes, "
              f"hasard = {base_rate:.2%})", "rappel", "précision")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pr_curves_validation.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------ importance des variables
    tree_family = max(("random_forest", "gbt"), key=lambda f: best[f]["row"]["auc_pr"])
    importances = best[tree_family]["model"].stages[-1].featureImportances.toArray()
    imp = (pd.Series(importances, index=cfg.MODEL_FEATURES)
             .sort_values(ascending=False).head(15)[::-1])
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh(imp.index, imp.values, color=COLORS[tree_family], height=0.7)
    style(ax, f"Importance des variables — {LABELS[tree_family]} (top 15)", "importance", "")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------ rapport
    lines = [
        "# Phase 4 — Comparaison des modèles supervisés\n",
        f"Entraînement : {n_train:,} transactions dont {n_train_fraud} fraudes. "
        f"Évaluation : validation ({n_valid_fraud} fraudes).\n",
        f"Poids de classe (variante « avec poids ») : normal = {w_normal:.3f}, "
        f"fraude = {w_fraud:.1f}.\n",
        "## Tous les essais (triés par AUC-PR)\n",
        "| Modèle | Réglage | Poids | AUC-PR | AUC-ROC | Durée |", "|---|---|---|---|---|---|",
    ]
    for r in res.sort_values("auc_pr", ascending=False).itertuples():
        lines.append(f"| {LABELS[r.family]} | {r.setting} | {r.weights} | {r.auc_pr:.4f} | "
                     f"{r.auc_roc:.4f} | {r.seconds:.0f} s |")

    lines += ["", "## Effet des poids de classe (meilleur réglage de chaque modèle)\n",
              "| Modèle | AUC-PR sans poids | AUC-PR avec poids |", "|---|---|---|"]
    effects = {"améliorent": [], "ne changent presque pas": [], "dégradent": []}
    for family in LABELS:
        sub = res[res.family == family]
        without = sub[sub.weights == "sans poids"].auc_pr.max()
        with_w = sub[sub.weights == "avec poids"].auc_pr.max()
        lines.append(f"| {LABELS[family]} | {without:.4f} | {with_w:.4f} |")
        # écart relatif de plus de 5 % = effet réel ; sinon « presque pas de changement »
        if with_w > without * 1.05:
            effects["améliorent"].append(LABELS[family])
        elif with_w < without * 0.95:
            effects["dégradent"].append(LABELS[family])
        else:
            effects["ne changent presque pas"].append(LABELS[family])
    lines.append("")
    lines.append("Sur ces données, les poids de classe " + " ; ".join(
        f"{verb} l'AUC-PR pour : {', '.join(models)}" for verb, models in effects.items() if models)
        + " (seuil : 5 % d'écart relatif).")

    lines += ["", "## Meilleur réglage par modèle (sauvegardé dans `models/`)\n",
              "| Modèle | Réglage | Poids | AUC-PR | AUC-ROC |", "|---|---|---|---|---|"]
    for family, b in sorted(best_json.items(), key=lambda kv: -kv[1]["auc_pr"]):
        lines.append(f"| {LABELS[family]} | {b['setting']} | {b['weights']} | "
                     f"{b['auc_pr']:.4f} | {b['auc_roc']:.4f} |")

    lines += ["", f"## Variables les plus importantes ({LABELS[tree_family]})\n",
              ", ".join(f"{n} ({v:.3f})" for n, v in imp[::-1].head(10).items()) + "\n",
              "## Remarque",
              f"La validation ne contient que {n_valid_fraud} fraudes : un écart de quelques "
              "centièmes d'AUC-PR entre deux modèles n'est pas significatif.\n",
              "## Graphiques",
              "- `figures/pr_curves_validation.png` — courbes précision-rappel",
              "- `figures/feature_importance.png` — importance des variables"]

    text = "\n".join(lines)
    (OUTPUT_DIR / "model_comparison.md").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
    spark.stop()


if __name__ == "__main__":
    main()
