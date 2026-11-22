# Validation-selected defaults

Successful CLI searches write their winner to `<dataset>/<model>.yaml` in this directory
by default. Ordinary training and the main-table launcher automatically use that pair's
saved model and training settings. No winner files are provided until a search is run;
without a saved file, training uses the original catalogs and training defaults.

Each file includes the data fingerprint and validation-selection provenance. It stores
model settings and training hyperparameters, not weights, test metrics, run directories,
devices, or the final training seed. A later successful search replaces the file for
the same pair atomically; other pairs remain unchanged. Failed searches leave it intact.

Use `tune --no-write-back` for an exploratory search that should not change defaults.
Use `train --ignore-tuned-config` to bypass saved settings. Both commands accept
`--tuned-config-dir` to use a different directory. Main-table runs use this directory.
Explicit Trainer overrides take priority over saved defaults. Searches themselves use
the base catalogs and explicit search overrides, independently of previously saved winners.

Saved defaults are rejected when their data fingerprint differs from the requested
dataset. Re-tune for the new data, select the matching saved directory, or explicitly
bypass saved defaults. The referenced original search files need not remain at the
same absolute path to use a saved configuration; their hash is retained for provenance.
