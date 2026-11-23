# Work journal

One entry per working session: objective, what was run, what was observed, how it is
interpreted, what was decided, and what comes next. Negative and inconclusive results
are recorded as well.

---

## 2026-11-24 — Step 1: reference measurements and reliable foundations

### Objective

Establish what the first version of the project actually measures, and put in place an
evaluation protocol on which later improvements can be compared without bias.

### What was run

```
python scripts/prepare_data.py
python scripts/run_baseline.py
pytest -q
```

Environment: Python 3.11, scikit-learn 1.8.0, 2 CPU cores. Full provenance (commit,
data checksum, library versions, configuration) is stored in each file of
`results/baseline/`.

### Observations

**Data.** The raw file holds 284,807 transactions and 492 frauds. It contains 1,081 rows
that are exact copies of an earlier row, 19 of them frauds. After de-duplication,
283,726 transactions and 473 frauds remain.

**Replication of the first version.** With the original protocol (random stratified
80/20 split, raw file) the random forest reproduces the published confusion matrix
exactly: `[[56859, 5], [18, 80]]`, i.e. precision 0.941, recall 0.816, F1 0.874 at a
threshold of 0.5. The AUPRC, which had not been reported, is 0.873 (95% CI
[0.808, 0.934]).

**Leakage in that protocol.** 344 rows of its test set are also present in its training
set, among them 8 of the 98 test frauds. In addition, the split is random although the
transactions are time-ordered, so the model is evaluated on transactions that precede
part of its training data.

**Same model, temporal protocol.** Trained on the first 80% of the de-duplicated
transactions (hours 0 to 40.3, 399 frauds) and evaluated on the last 20% (hours 40.3 to
48.0, 74 frauds), the same model reaches precision 0.962, recall 0.689, F1 0.803 and an
AUPRC of 0.784 (95% CI [0.689, 0.869]).

**Cross-validation on the development period.** Expanding-window cross-validation with
four folds (47 to 89 frauds per validation fold):

| Feature set | AUPRC per fold | Mean (std) |
|---|---|---|
| raw elapsed time | 0.809, 0.654, 0.806, 0.737 | 0.752 (0.073) |
| cyclic time of day | 0.803, 0.638, 0.813, 0.748 | 0.751 (0.080) |

### Interpretation

- The figures of the first version are reproducible but optimistic. Under a protocol
  that respects chronology and removes duplicates, recall at the default threshold falls
  from 0.82 to 0.69 and AUPRC from 0.87 to 0.78.
- This difference must be read with caution: the two test sets differ (98 and 74
  frauds), the confidence intervals overlap, and a single seed was used. It indicates
  the direction of the bias; it does not quantify it precisely.
- Performance varies markedly between periods (fold AUPRC from 0.65 to 0.81). With so
  few frauds per fold, differences between models smaller than this spread cannot be
  regarded as established.
- Replacing the elapsed time by the time of day does not change performance (0.751
  against 0.752). The substitution is retained for validity, not for accuracy: the
  elapsed time is undefined for any transaction recorded after the collection period.

### Decisions

Recorded in `docs/DECISIONS.md` (D1).

### Note on the hold-out set

The temporal hold-out set was evaluated once in this step, with the unchanged model of
the first version, to measure the effect of the protocol. No modelling choice was or
will be based on that figure. From this point on, every choice (model family,
hyper-parameters, features, calibration, threshold) relies exclusively on
cross-validation over the development period; the hold-out set is evaluated again only
once, at the end, for the retained configuration.

### State of the repository after this step

- The code is an installable package (`src/fraud_detection`) driven by
  `configs/config.yaml`.
- 24 unit tests, including leakage guards on every split, run in continuous
  integration.
- The serving layer (`api/`, `dashboard/`) is unchanged and still expects the model
  files of the first version; it is rewritten in a later step. The first version remains
  available under the tag `v0-legacy`.

### Next

Step 2: logistic regression and gradient-boosted trees, hyper-parameter search under a
fixed budget, treatment of class imbalance and probability calibration, all compared by
cross-validation on the development period.
