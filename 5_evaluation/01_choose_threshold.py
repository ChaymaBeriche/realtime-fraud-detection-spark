"""
Phase 5 — Evaluation
Script 01 : choix du seuil de décision, sur la VALIDATION uniquement.

Le modèle donne à chaque transaction une probabilité de fraude ; le seuil décide à
partir de quelle probabilité on lève une alerte.

Règle retenue : maximiser le F2 (F-beta avec beta = 2). Le F2 compte le rappel deux
fois plus que la précision : une fraude manquée est jugée plus grave qu'une fausse
alerte. Aucune valeur arbitraire (coût, devise) n'intervient dans le choix.

Le seuil est placé au milieu entre le score de la dernière alerte retenue et le score
suivant, pour garder une marge sur des données nouvelles.

Le rapport montre aussi :
  - le compromis précision / rappel pour plusieurs seuils
  - la stabilité du choix (plage de seuils dont le F2 reste proche du maximum)
  - les montants de fraude détectés et manqués (vue métier, à titre d'information)

Entrées : models/<modèle retenu en phase 4>, data/prepared/validation
Sorties : models/decision.json                     (modèle + seuil, utilisé en phase 6)
          5_evaluation/outputs/threshold_report.md
          5_evaluation/outputs/figures/threshold_validation.png

Usage (depuis la racine du projet) :
    python 5_evaluation/01_choose_threshold.py
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

import common as cm
import config as cfg

OUTPUT_DIR = cfg.EVALUATION_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
GRID_THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
STABLE_SHARE = 0.95          # « stable » : F2 >= 95 % du F2 maximal


def main():
    spark = cfg.spark_session("FraudDetection-Threshold")
    family = cm.selected_family()
    _, val = cm.score_split(spark, family, "validation")
    y = val[cfg.LABEL_COL].to_numpy()
    s = val["score"].to_numpy()
    amount = val["Amount"].to_numpy()
    n_fraud = int(y.sum())
    hours = (val["Time"].max() - val["Time"].min()) / 3600

    # ------------------------------------------------------------ 1. seuil qui maximise F2
    thr, prec, rec, f2 = cm.curve_by_threshold(y, s)
    i = int(np.argmax(f2))
    next_score = thr[i + 1] if i + 1 < len(thr) else 0.0
    threshold = float((thr[i] + next_score) / 2)       # milieu : marge pour les données futures
    chosen = cm.metrics_at(y, s, amount, threshold)

    # ------------------------------------------------------------ 2. stabilité du choix
    stable = thr[f2 >= STABLE_SHARE * f2[i]]
    stable_lo, stable_hi = float(stable.min()), float(stable.max())

    # ------------------------------------------------------------ 3. tableau de compromis
    rows = [cm.metrics_at(y, s, amount, t) for t in GRID_THRESHOLDS]
    total_fraud_amount = float(amount[y == 1].sum())

    # ------------------------------------------------------------ 4. décision pour la suite
    decision = {
        "model": family,
        "model_path": cm.best_models()[family]["path"],
        "threshold": round(threshold, 6),
        "rule": f"maximisation du F{cfg.F_BETA:g}",
        "validation": {k: chosen[k] for k in ("alerts", "tp", "fp", "fn", "precision", "recall",
                                              "f1", "f2")},
    }
    cfg.DECISION_FILE.write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------ 5. graphique
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(0.0, 1.0, 201)
    # sans aucune alerte, la précision n'est pas définie : la courbe s'arrête là
    curves = np.array([[m["precision"] if m["alerts"] else np.nan, m["recall"], m["f2"]]
                       for m in (cm.metrics_at(y, s, amount, t) for t in grid)])
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(grid, curves[:, 0], color=cm.C_BLUE, linewidth=2, label="précision")
    ax.plot(grid, curves[:, 1], color=cm.C_ORANGE, linewidth=2, label="rappel")
    ax.plot(grid, curves[:, 2], color=cm.C_AQUA, linewidth=2, label="F2")
    ax.axvspan(stable_lo, stable_hi, color=cm.GRID, alpha=0.6, linewidth=0)
    ax.axvline(threshold, color=cm.INK_2, linewidth=1, linestyle="--")
    ax.text(threshold, 1.03, f"seuil {threshold:.3f}", ha="center", color=cm.INK, fontsize=9)
    cm.style(ax, f"{cm.LABELS[family]} — choix du seuil (validation, {n_fraud} fraudes)",
             "seuil de décision", "valeur")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "threshold_validation.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------ 6. rapport
    width = stable_hi - stable_lo
    stability = ("**stable** : le F2 reste proche du maximum sur une plage large de seuils"
                 if width >= 0.10 else
                 "**sensible** : le F2 ne reste proche du maximum que sur une plage étroite ; "
                 "avec peu de fraudes en validation, le seuil exact est incertain")
    lines = [
        "# Phase 5 — Choix du seuil de décision (validation)\n",
        f"Modèle : **{cm.LABELS[family]}** (retenu en phase 4). Validation : {len(y):,} "
        f"transactions, {n_fraud} fraudes.\n",
        f"Règle : maximiser le F{cfg.F_BETA:g}, qui compte le rappel deux fois plus que la précision.\n",
        "## 1. Seuil retenu\n",
        f"**Seuil = {threshold:.4f}** : une alerte est levée si la probabilité de fraude ≥ {threshold:.4f}.\n",
        "| Alertes | Fraudes détectées | Fausses alertes | Fraudes manquées | Précision | Rappel | F1 | F2 |",
        "|---|---|---|---|---|---|---|---|",
        f"| {chosen['alerts']} | {chosen['tp']} | {chosen['fp']} | {chosen['fn']} | "
        f"{chosen['precision']:.3f} | {chosen['recall']:.3f} | {chosen['f1']:.3f} | {chosen['f2']:.3f} |",
        "",
        "## 2. Stabilité du choix\n",
        f"Le F2 reste ≥ {STABLE_SHARE:.0%} de son maximum pour les seuils entre "
        f"**{stable_lo:.3f}** et **{stable_hi:.3f}** (zone grisée du graphique). "
        f"Le choix est {stability}.\n",
        "## 3. Compromis précision / rappel selon le seuil\n",
        "| Seuil | Alertes | Fraudes détectées | Fausses alertes | Précision | Rappel | F2 | "
        "Montant détecté | Montant manqué |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows + [chosen]:
        mark = " **(retenu)**" if r is chosen else ""
        lines.append(f"| {r['threshold']:.3f}{mark} | {r['alerts']} | {r['tp']} | {r['fp']} | "
                     f"{r['precision']:.3f} | {r['recall']:.3f} | {r['f2']:.3f} | "
                     f"{cm.fmt_amount(r['amount_caught'])} | {cm.fmt_amount(r['amount_missed'])} |")
    lines += [
        "",
        f"Montant total des fraudes de la validation : {cm.fmt_amount(total_fraud_amount)} "
        "(devise non précisée par la source).\n",
        "## 4. Vue métier (au seuil retenu)\n",
        f"- {chosen['alerts']} alertes à vérifier sur {len(y):,} transactions "
        f"({chosen['alerts'] / len(y):.2%}), soit environ {chosen['alerts'] / hours:.0f} par heure "
        f"(la validation couvre {hours:.1f} heures).",
        f"- {cm.fmt_amount(chosen['amount_caught'])} de fraude détectée, "
        f"{cm.fmt_amount(chosen['amount_missed'])} manquée "
        f"(soit {chosen['amount_caught'] / total_fraud_amount:.0%} du montant total des fraudes).",
        "",
        "## Fichiers",
        "- `models/decision.json` — modèle et seuil retenus (utilisés en phase 6)",
        "- `figures/threshold_validation.png` — précision, rappel et F2 selon le seuil",
    ]
    text = "\n".join(lines)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "threshold_report.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    spark.stop()


if __name__ == "__main__":
    main()
