"""
Phase 5 — fonctions communes aux deux scripts d'évaluation.

Les scores sont calculés par Spark (pipeline MLlib), puis seules cinq colonnes
(Time, Amount, hour, Class, score) sont ramenées en pandas : ~55 000 lignes, ce qui permet
de calculer précisément les métriques pour chaque seuil possible.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array

import config as cfg

LABELS = {"logistic_regression": "Régression logistique", "random_forest": "Random Forest",
          "gbt": "GBT"}
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def best_models():
    path = cfg.MODELING_DIR / "outputs" / "best_models.json"
    return json.loads(path.read_text(encoding="utf-8"))


def selected_family():
    """Modèle retenu en phase 4 : meilleure AUC-PR sur la validation."""
    bm = best_models()
    return max(bm, key=lambda f: bm[f]["auc_pr"])


def score_split(spark, family, split):
    """Applique le pipeline sauvegardé à une partie (validation / test).
    Retourne (DataFrame Spark des prédictions, DataFrame pandas Amount/hour/Class/score)."""
    model = PipelineModel.load(str(cfg.ROOT / best_models()[family]["path"]))
    df = spark.read.parquet(str(cfg.PREPARED_DIR / split))
    pred = model.transform(df).withColumn("score", vector_to_array("probability")[1])
    pdf = pred.select("Time", "Amount", "hour", cfg.LABEL_COL, "score").toPandas()
    return pred, pdf


def fbeta(precision, recall, beta=cfg.F_BETA):
    b2 = beta * beta
    denom = b2 * precision + recall
    return np.where(denom > 0, (1 + b2) * precision * recall / np.where(denom > 0, denom, 1), 0.0)


def metrics_at(y, score, amount, threshold):
    """Toutes les mesures utiles pour un seuil donné (alerte si score >= seuil)."""
    alert = score >= threshold
    fraud = y == 1
    tp = int(np.sum(alert & fraud))
    fp = int(np.sum(alert & ~fraud))
    fn = int(np.sum(~alert & fraud))
    tn = int(np.sum(~alert & ~fraud))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(threshold=float(threshold), alerts=tp + fp, tp=tp, fp=fp, fn=fn, tn=tn,
                precision=precision, recall=recall, f1=f1,
                f2=float(fbeta(np.array(precision), np.array(recall))),
                amount_caught=float(amount[alert & fraud].sum()),
                amount_missed=float(amount[~alert & fraud].sum()))


def curve_by_threshold(y, score):
    """Précision, rappel et F2 pour chaque seuil distinct (calcul vectorisé).
    Retourne les seuils (scores distincts, décroissants) et les mesures associées."""
    order = np.argsort(-score, kind="mergesort")
    s, yy = score[order], y[order]
    tp = np.cumsum(yy)
    fp = np.cumsum(1 - yy)
    last = np.r_[s[1:] != s[:-1], True]          # dernière position de chaque score distinct
    thr, tp, fp = s[last], tp[last], fp[last]
    precision = tp / (tp + fp)
    recall = tp / yy.sum()
    return thr, precision, recall, fbeta(precision, recall)


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2)


def fmt_amount(x):
    return f"{x:,.2f}".replace(",", " ")
