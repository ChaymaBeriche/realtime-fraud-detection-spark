"""
Phase 5 — Evaluation
Script 02 : évaluation finale sur le TEST, avec le modèle et le seuil choisis.

Le test n'a servi à aucun choix (ni modèle, ni réglage, ni seuil) : son score est
une estimation honnête de ce que donnerait le système sur des transactions futures.
Ce script ne doit être lancé qu'APRÈS le script 01, et ses résultats ne doivent pas
servir à revenir sur les choix précédents.

Contenu :
  1. matrice de confusion et métriques au seuil retenu (précision, rappel, F1, F2)
  2. AUC-PR et AUC-ROC (indépendantes du seuil)
  3. montants de fraude détectés et manqués
  4. intervalles de confiance à 95 % par bootstrap (le test ne contient que 70 fraudes)
  5. comparaison validation / test
  6. analyse des erreurs : fraudes manquées et fausses alertes
  7. pour information : régression logistique et GBT sur le test (AUC seulement)

Entrées : models/decision.json (script 01), models/*, data/prepared/test
Sorties : 5_evaluation/outputs/test_report.md
          5_evaluation/outputs/figures/test_pr_curve.png
          5_evaluation/outputs/figures/test_scores.png

Usage (depuis la racine du projet) :
    python 5_evaluation/02_evaluate_test.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyspark.ml.evaluation import BinaryClassificationEvaluator

import common as cm
import config as cfg

OUTPUT_DIR = cfg.EVALUATION_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"


def spark_aucs(pred):
    """AUC-PR et AUC-ROC calculées par Spark (même méthode qu'en phase 4)."""
    ev = lambda m: BinaryClassificationEvaluator(labelCol=cfg.LABEL_COL, rawPredictionCol="probability",
                                                 metricName=m).evaluate(pred)
    return ev("areaUnderPR"), ev("areaUnderROC")


def bootstrap(y, s, amount, threshold, n_samples, seed):
    """Intervalles de confiance à 95 % : on rééchantillonne le test avec remise
    n_samples fois et on recalcule les métriques à chaque fois."""
    rng = np.random.default_rng(seed)
    n = len(y)
    alert = s >= threshold
    fraud = y == 1
    out = {"precision": [], "recall": [], "f2": [], "amount_share": []}
    for _ in range(n_samples):
        idx = rng.integers(0, n, n)
        a, f, am = alert[idx], fraud[idx], amount[idx]
        tp, fp, fn = np.sum(a & f), np.sum(a & ~f), np.sum(~a & f)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        out["precision"].append(p)
        out["recall"].append(r)
        out["f2"].append(float(cm.fbeta(np.array(p), np.array(r))))
        total = am[f].sum()
        out["amount_share"].append(am[a & f].sum() / total if total else 0.0)
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in out.items()}


def main():
    decision = json.loads(cfg.DECISION_FILE.read_text(encoding="utf-8"))
    family, threshold = decision["model"], decision["threshold"]

    spark = cfg.spark_session("FraudDetection-Test")
    pred, test = cm.score_split(spark, family, "test")
    y = test[cfg.LABEL_COL].to_numpy()
    s = test["score"].to_numpy()
    amount = test["Amount"].to_numpy()
    hour = test["hour"].to_numpy()
    n, n_fraud = len(y), int(y.sum())
    hours = (test["Time"].max() - test["Time"].min()) / 3600

    # ------------------------------------------------------------ 1-3. métriques
    m = cm.metrics_at(y, s, amount, threshold)
    auc_pr, auc_roc = spark_aucs(pred)
    total_amount = float(amount[y == 1].sum())
    amount_share = m["amount_caught"] / total_amount if total_amount else 0.0

    # ------------------------------------------------------------ 4. bootstrap
    ci = bootstrap(y, s, amount, threshold, cfg.BOOTSTRAP_SAMPLES, cfg.RANDOM_SEED)

    # ------------------------------------------------------------ 6. analyse des erreurs
    alert = s >= threshold
    fraud = y == 1
    groups = {
        "Fraudes détectées": alert & fraud,
        "Fraudes manquées": ~alert & fraud,
        "Fausses alertes": alert & ~fraud,
        "Transactions normales (toutes)": ~fraud,
    }

    def describe(mask):
        if not mask.any():
            return dict(n=0, med_amount=float("nan"), med_score=float("nan"), zero=float("nan"))
        return dict(n=int(mask.sum()), med_amount=float(np.median(amount[mask])),
                    med_score=float(np.median(s[mask])), zero=float(np.mean(amount[mask] == 0)))

    desc = {k: describe(v) for k, v in groups.items()}
    missed = test[~alert & fraud].sort_values("score", ascending=False)

    # ------------------------------------------------------------ 7. autres modèles (information)
    others = []
    for fam in cm.best_models():
        if fam == family:
            continue
        p_other, _ = cm.score_split(spark, fam, "test")
        others.append((fam, *spark_aucs(p_other)))

    # ------------------------------------------------------------ graphiques
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    thr, prec, rec, _ = cm.curve_by_threshold(y, s)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(rec, prec, color=cm.C_ORANGE, linewidth=2,
            label=f"{cm.LABELS[family]} (AUC-PR {auc_pr:.3f})")
    ax.scatter([m["recall"]], [m["precision"]], s=70, color=cm.INK, zorder=3,
               edgecolor="white", linewidth=2)
    ax.annotate(f"seuil {threshold:.3f}\nprécision {m['precision']:.2f}, rappel {m['recall']:.2f}",
                (m["recall"], m["precision"]), xytext=(-12, -12), textcoords="offset points",
                ha="right", va="top", color=cm.INK, fontsize=9)
    cm.style(ax, f"Courbe précision-rappel — test ({n_fraud} fraudes)", "rappel", "précision")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "test_pr_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.linspace(0, 1, 41)
    for mask, color, label in [(~fraud, cm.C_BLUE, "Normal"), (fraud, cm.C_ORANGE, "Fraude")]:
        ax.hist(s[mask], bins=bins, density=True, histtype="step", linewidth=2, color=color, label=label)
    ax.axvline(threshold, color=cm.INK_2, linewidth=1, linestyle="--")
    ax.set_yscale("log")
    cm.style(ax, "Probabilité de fraude donnée par le modèle — test", "score", "densité (échelle log)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "test_scores.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------ rapport
    v = decision["validation"]
    pr_drop = v["recall"] - m["recall"]
    gap_text = ("Le rappel sur le test est **proche** de celui de la validation : le seuil se "
                "généralise bien." if abs(pr_drop) <= 0.10 else
                ("Le rappel **baisse** nettement sur le test par rapport à la validation."
                 if pr_drop > 0 else "Le rappel est **meilleur** sur le test que sur la validation."))
    lo_r, hi_r = ci["recall"]

    lines = [
        "# Phase 5 — Évaluation finale sur le test\n",
        f"Modèle : **{cm.LABELS[family]}**, seuil **{threshold:.4f}** (choisi sur la validation "
        f"par {decision['rule']}).",
        f"Test : {n:,} transactions sur {hours:.1f} heures, {n_fraud} fraudes. "
        "Le test n'a servi à aucun choix.\n",
        "## 1. Matrice de confusion\n",
        "| | Prédit fraude (alerte) | Prédit normal |", "|---|---|---|",
        f"| **Vraie fraude** | {m['tp']} (détectées) | {m['fn']} (manquées) |",
        f"| **Vraie normale** | {m['fp']} (fausses alertes) | {m['tn']:,} |",
        "",
        "## 2. Métriques (intervalle de confiance à 95 %, bootstrap "
        f"{cfg.BOOTSTRAP_SAMPLES} rééchantillonnages)\n",
        "| Métrique | Valeur | Intervalle à 95 % |", "|---|---|---|",
        f"| Précision | {m['precision']:.3f} | [{ci['precision'][0]:.3f} ; {ci['precision'][1]:.3f}] |",
        f"| Rappel | {m['recall']:.3f} | [{lo_r:.3f} ; {hi_r:.3f}] |",
        f"| F1 | {m['f1']:.3f} | — |",
        f"| F2 | {m['f2']:.3f} | [{ci['f2'][0]:.3f} ; {ci['f2'][1]:.3f}] |",
        f"| AUC-PR | {auc_pr:.3f} | — |",
        f"| AUC-ROC | {auc_roc:.3f} | — |",
        "",
        f"Avec {n_fraud} fraudes seulement, le rappel réel se situe vraisemblablement entre "
        f"{lo_r:.0%} et {hi_r:.0%}.\n",
        "## 3. Vue métier\n",
        f"- {m['alerts']} alertes sur {n:,} transactions ({m['alerts'] / n:.2%}), soit environ "
        f"{m['alerts'] / hours:.0f} par heure ; {m['precision']:.0%} d'entre elles sont de vraies fraudes.",
        f"- Montant des fraudes : {cm.fmt_amount(total_amount)} au total, "
        f"**{cm.fmt_amount(m['amount_caught'])} détecté ({amount_share:.0%})**, "
        f"{cm.fmt_amount(m['amount_missed'])} manqué. Intervalle à 95 % de la part détectée : "
        f"[{ci['amount_share'][0]:.0%} ; {ci['amount_share'][1]:.0%}].",
        "",
        "## 4. Validation et test\n",
        "| | Alertes | Fraudes détectées | Fausses alertes | Précision | Rappel | F2 |",
        "|---|---|---|---|---|---|---|",
        f"| Validation | {v['alerts']} | {v['tp']} | {v['fp']} | {v['precision']:.3f} | "
        f"{v['recall']:.3f} | {v['f2']:.3f} |",
        f"| Test | {m['alerts']} | {m['tp']} | {m['fp']} | {m['precision']:.3f} | "
        f"{m['recall']:.3f} | {m['f2']:.3f} |",
        "",
        gap_text + "\n",
        "## 5. Analyse des erreurs\n",
        "| Groupe | Nombre | Montant médian | Score médian | Part de montants nuls |",
        "|---|---|---|---|---|",
    ]
    for name, d in desc.items():
        if d["n"] == 0:
            lines.append(f"| {name} | 0 | — | — | — |")
        else:
            lines.append(f"| {name} | {d['n']:,} | {d['med_amount']:.2f} | {d['med_score']:.3f} | "
                         f"{d['zero']:.0%} |")
    lines += ["", f"Les {len(missed)} fraudes manquées (triées par score) :\n",
              "| Montant | Heure relative | Score |", "|---|---|---|"]
    for r in missed.itertuples():
        lines.append(f"| {r.Amount:.2f} | {int(r.hour)} | {r.score:.3f} |")

    d_missed, d_caught = desc["Fraudes manquées"], desc["Fraudes détectées"]
    if d_missed["n"]:
        lines += ["",
                  f"Les fraudes manquées ont un score médian de {d_missed['med_score']:.3f}, contre "
                  f"{d_caught['med_score']:.3f} pour les fraudes détectées : pour le modèle, elles "
                  "ressemblent à des transactions normales. "
                  + (f"Leur montant médian ({d_missed['med_amount']:.2f}) est "
                     + ("plus faible" if d_missed["med_amount"] < d_caught["med_amount"] else "plus élevé")
                     + f" que celui des fraudes détectées ({d_caught['med_amount']:.2f}).")]
    lines += ["",
              "Rappel : le test couvre les heures relatives 16 à 23 seulement (aucune heure de nuit, "
              "voir phase 3).\n",
              "## 6. Pour information : autres modèles sur le test\n",
              "Le choix du modèle a été fait sur la validation ; ces chiffres ne servent pas à le "
              "modifier.\n",
              "| Modèle | AUC-PR test | AUC-ROC test |", "|---|---|---|",
              f"| {cm.LABELS[family]} (retenu) | {auc_pr:.3f} | {auc_roc:.3f} |"]
    for fam, a_pr, a_roc in others:
        lines.append(f"| {cm.LABELS[fam]} | {a_pr:.3f} | {a_roc:.3f} |")
    lines += ["",
              "## Graphiques",
              "- `figures/test_pr_curve.png` — courbe précision-rappel et point de fonctionnement",
              "- `figures/test_scores.png` — distribution des scores (normal / fraude) et seuil"]

    text = "\n".join(lines)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "test_report.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    spark.stop()


if __name__ == "__main__":
    main()
