"""
Phase 6 — Deployment
Script 01 : simulateur de flux de transactions (le « producteur »).

Rejoue les transactions du TEST dans l'ordre du temps, en accéléré, comme si elles
arrivaient en direct d'un système de paiement :
  - vitesse x60 par défaut : 1 seconde réelle = 1 minute de transactions
    (les 7,2 heures du test défilent en ~7 minutes)
  - chaque seconde, un petit fichier CSV contenant les transactions de cette minute
    est déposé dans data/stream/input/ ; Spark (script 02) surveille ce dossier
  - le fichier est d'abord écrit sous un nom caché puis renommé : Spark ne voit
    jamais un fichier à moitié écrit
  - chaque fichier contient seulement ce qu'un système de paiement connaîtrait
    (identifiant, Time, Amount, V1..V28) + l'heure d'envoi (pour mesurer la latence)

La vraie classe (Class) n'est PAS envoyée dans le flux : elle est écrite à part dans
data/stream/labels.csv et ne sert qu'au contrôle final (script 03). Dans la réalité,
on ne sait qu'une transaction était frauduleuse que bien plus tard.

Le simulateur attend que le script 02 soit prêt (fichier data/stream/_READY) : on peut
donc lancer les deux scripts dans n'importe quel ordre. À la fin, un fichier « _DONE »
signale au script 02 que le flux est terminé.

Ce script n'utilise pas Spark (il joue le rôle d'un système extérieur).

Usage (depuis la racine du projet) :
    python 6_deployment/01_stream_simulator.py            # vitesse x60
    python 6_deployment/01_stream_simulator.py --speed 600   # x600, ~45 secondes
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config as cfg

SENT_COLUMNS = ["transaction_id", "Time", "Amount"] + cfg.PCA_COLS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=cfg.STREAM_SPEED,
                        help="facteur d'accélération (60 = 1 s réelle pour 1 min de données)")
    args = parser.parse_args()

    # ------------------------------------------------------------ données à rejouer
    test = pd.read_parquet(cfg.PREPARED_DIR / "test")
    test = test.sort_values("Time", kind="mergesort").reset_index(drop=True)
    test.insert(0, "transaction_id", range(len(test)))

    # ------------------------------------------------------------ attendre la détection
    if not cfg.STREAM_READY_FLAG.exists():
        print("En attente du script de détection (6_deployment/02_stream_detection.py)...")
        while not cfg.STREAM_READY_FLAG.exists():
            time.sleep(1)
    print("Détection prête.")

    # ------------------------------------------------------------ dossier du flux, vidé
    cfg.STREAM_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in cfg.STREAM_INPUT_DIR.iterdir():          # on vide le contenu, pas le dossier
        f.unlink()                                    # (Spark le surveille déjà)
    test[["transaction_id", cfg.LABEL_COL]].to_csv(cfg.STREAM_LABELS_FILE, index=False)

    t_first, t_last = test["Time"].min(), test["Time"].max()
    n_ticks = int((t_last - t_first) // args.speed) + 1
    print(f"Rejeu de {len(test):,} transactions ({(t_last - t_first) / 3600:.1f} h de données) "
          f"à la vitesse x{args.speed:g} : environ {n_ticks / 60:.1f} minutes.")
    print(f"Dossier du flux : {cfg.STREAM_INPUT_DIR}")

    # ------------------------------------------------------------ rejeu
    tick_of = ((test["Time"] - t_first) // args.speed).astype(int)
    start = time.time()
    sent = files = 0
    for tick, batch in test.groupby(tick_of, sort=True):
        # attendre le moment où ces transactions « arrivent »
        delay = start + tick + 1 - time.time()
        if delay > 0:
            time.sleep(delay)
        out = batch[SENT_COLUMNS].copy()
        out["sent_at"] = time.time()                          # heure d'envoi (secondes)
        name = f"tx_{tick:05d}.csv"
        tmp = cfg.STREAM_INPUT_DIR / f".{name}.tmp"           # nom caché : ignoré par Spark
        out.to_csv(tmp, index=False)
        os.replace(tmp, cfg.STREAM_INPUT_DIR / name)          # renommage atomique
        sent += len(out)
        files += 1
        if files % 60 == 0:
            print(f"  {sent:>6,} transactions envoyées ({files} fichiers, "
                  f"{(time.time() - start) / 60:.1f} min)")

    cfg.STREAM_DONE_FLAG.write_text(f"{sent}\n", encoding="utf-8")
    cfg.STREAM_READY_FLAG.unlink(missing_ok=True)     # la prochaine exécution attendra une détection neuve
    print(f"Terminé : {sent:,} transactions en {files} fichiers, "
          f"{(time.time() - start) / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
