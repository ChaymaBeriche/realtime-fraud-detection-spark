# Rapport de validation du dataset

Fichier : `data/archive/creditcard.csv` (150,828,752 octets)

| Contrôle | Résultat | Attendu |
|---|---|---|
| Lignes | 284,807 | 284,807 |
| Fraudes | 492 (0.173%) | 492 |
| Colonnes | 31 | 31 |
| Valeurs manquantes | 0 | 0 |
| Doublons exacts | 1,081 | — (traités en phase 3) |
| Période couverte | 48.0 heures | ~48 heures |

**Verdict : conforme au dataset officiel.**

## Schéma utilisé

```
root
 |-- Time: double (nullable = true)
 |-- V1: double (nullable = true)
 |-- V2: double (nullable = true)
 |-- V3: double (nullable = true)
 |-- V4: double (nullable = true)
 |-- V5: double (nullable = true)
 |-- V6: double (nullable = true)
 |-- V7: double (nullable = true)
 |-- V8: double (nullable = true)
 |-- V9: double (nullable = true)
 |-- V10: double (nullable = true)
 |-- V11: double (nullable = true)
 |-- V12: double (nullable = true)
 |-- V13: double (nullable = true)
 |-- V14: double (nullable = true)
 |-- V15: double (nullable = true)
 |-- V16: double (nullable = true)
 |-- V17: double (nullable = true)
 |-- V18: double (nullable = true)
 |-- V19: double (nullable = true)
 |-- V20: double (nullable = true)
 |-- V21: double (nullable = true)
 |-- V22: double (nullable = true)
 |-- V23: double (nullable = true)
 |-- V24: double (nullable = true)
 |-- V25: double (nullable = true)
 |-- V26: double (nullable = true)
 |-- V27: double (nullable = true)
 |-- V28: double (nullable = true)
 |-- Amount: double (nullable = true)
 |-- Class: integer (nullable = true)
```

Copie Parquet écrite dans `data/parquet/creditcard/`.
