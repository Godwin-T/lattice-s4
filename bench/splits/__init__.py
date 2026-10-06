"""
Splits: a canonical job table -> one frozen folds manifest.

This package reads a canonical table and writes `folds.json`. It contains no
model, no features and no metrics; only the train/test split that every arm must
share. See `splits.md` for the plan and `folds_manifest.md` for the contract.
"""
