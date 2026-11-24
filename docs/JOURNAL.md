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

---

## 2026-11-24 — Step 2: model selection

### Objective

Determine whether another model family improves on the random forest of the first
version, whether the class imbalance should be corrected, and how to obtain usable
probabilities. Every comparison relies on expanding-window cross-validation over the
development period; the hold-out set is not read.

### What was run

```
python scripts/run_model_selection.py
python scripts/update_readme.py
```

Three stages: random search of hyper-parameters (20 configurations for logistic
regression, 30 each for LightGBM and XGBoost, one seed); comparison of four imbalance
treatments for the best configuration of each family (five seeds); comparison of three
calibration methods (five seeds). Results are in `results/model_selection/`.

### Observations

**Model families (no imbalance treatment, mean over five seeds).**

| Model | AUPRC | Per validation period | Recall at FPR ≤ 0.1% |
|---|---|---|---|
| Random forest (first version) | 0.757 | 0.804, 0.652, 0.811, 0.762 | 0.803 |
| Logistic regression | 0.749 | 0.741, 0.628, 0.859, 0.768 | 0.792 |
| LightGBM | 0.778 | 0.814, 0.697, 0.817, 0.786 | 0.803 |
| XGBoost | 0.780 | 0.812, 0.703, 0.820, 0.786 | 0.795 |

- Both boosted models exceed the random forest on each of the four periods and for each
  of the five seeds, by about 0.02 on average.
- LightGBM and XGBoost differ by 0.002, less than their variation between seeds
  (standard deviation 0.002 to 0.003).
- Logistic regression matches the random forest only under strong regularisation
  (C ≈ 2 × 10⁻⁴); with weak regularisation its AUPRC falls to 0.54.
- The recall at a false-positive rate of 0.1% is not improved by any model.

**Sensitivity to settings.** Over the 30 configurations searched, the AUPRC of XGBoost
ranges from 0.686 to 0.783 and that of LightGBM from 0.588 to 0.781.

**Imbalance treatments.** For XGBoost: none 0.780, SMOTE 0.780, class weighting 0.775,
under-sampling 0.747. For LightGBM: none 0.778, SMOTE 0.779, under-sampling 0.756, class
weighting 0.616. No treatment improves on the untreated model beyond seed-to-seed
variability.

**Calibration (XGBoost).** Brier score 0.000522 without calibration, 0.000483 with
sigmoid calibration, 0.000554 with isotonic calibration. AUPRC is unchanged by the
sigmoid (0.772) and reduced by the isotonic method (0.726).

### Interpretation

- The gain of boosted trees over the random forest is small compared with the variation
  between periods (0.65 to 0.82), but it is consistent across periods and seeds, which a
  paired comparison is able to show.
- Accuracy does not separate XGBoost from LightGBM on this dataset. Robustness to the
  settings does: LightGBM has the wider spread over the search and collapses under class
  weighting.
- Re-sampling brings no benefit here and distorts the class proportions seen by the
  model, hence the predicted probabilities.
- Isotonic calibration is fitted on the last fifth of each training window, which holds
  a few tens of frauds; its step function creates ties that degrade the ranking.

### Correction made during this step

The first version of the selection script retained, for each family, the treatment with
the highest mean AUPRC. It selected SMOTE on the strength of a difference of 0.001. The
rule was replaced: a treatment is retained only if it exceeds the absence of treatment
by more than twice the standard deviation over seeds. The search range of the
regularisation strength of the logistic regression was also extended after the best
value of the first search fell on its lower bound.

### Caveats

- The reported cross-validated scores are those of the best of 20 or 30 configurations
  and are therefore optimistically biased; the unbiased estimate will come from the
  hold-out set at the end.
- The hyper-parameter search uses a single seed; only the retained configurations were
  repeated over five seeds.
- Averaging several models and longer searches were not explored.

### Decisions

Recorded in `docs/DECISIONS.md` (D2): XGBoost, no imbalance treatment, sigmoid
calibration.

### Next

Step 3: behavioural features on a simulated dataset with card and terminal identifiers.

---

## 2026-11-24 — Step 3: behavioural features on simulated transactions

### Objective

Build features that describe the habits of a customer and the recent history of a
terminal, guarantee that they use only information available at the time of each
transaction, and measure what each group of features contributes. The main dataset has
no identifier; the study is carried out on simulated transactions, as decided in D1.

### What was run

```
python scripts/run_behavioural_features.py
python scripts/update_readme.py
```

Results are in `results/behavioural/`.

### Data

The simulator (`fraud_detection.simulation`) is an independent implementation of the
generative process specified in the *Fraud Detection Handbook*, with its baseline
settings: 5,000 customers, 10,000 terminals, 183 days. It produces 1,793,343
transactions, 15,268 of them frauds (0.85%): 1,109 by amount above the threshold, 9,268
on compromised terminals and 4,891 on compromised customers. The handbook reports
1,754,155 transactions and 14,681 frauds (0.84%) for its own random draws; the two
tables are comparable, not identical.

### Features

- Raw: amount, cyclic time of day.
- Calendar: weekend flag, night flag.
- Customer, over 1, 7 and 30 days: number of transactions, mean amount, ratio of the
  current amount to that mean. The window includes the current transaction.
- Terminal, over 1, 7 and 30 days: number of transactions and share of frauds, both
  shifted by a label delay of 7 days.

Three tests establish point-in-time correctness: the features of past transactions are
unchanged when all later transactions are removed; the terminal risk is unchanged when
every label of the last days, within the delay, is altered; no feature changes when the
label of the current transaction changes.

### Protocol

The last 20% of the period is set aside as a hold-out set and not read. On the remaining
146 days, four expanding-window folds are used; each training window ends 7 days before
its validation block, since the labels of the most recent days would not be known at
training time. The model is the XGBoost configuration retained in D2, without
re-tuning, over three seeds.

### Observations

| Feature set | AUPRC | Recall at FPR ≤ 0.1% | Amount scenario | Terminal scenario | Customer scenario |
|---|---|---|---|---|---|
| Raw | 0.164 | 0.139 | 0.541 | 0.001 | 0.324 |
| Raw + calendar | 0.163 | 0.139 | 0.545 | 0.001 | 0.324 |
| Raw + customer | 0.349 | 0.323 | 0.976 | 0.002 | 0.819 |
| Raw + terminal | 0.590 | 0.419 | 0.275 | 0.559 | 0.171 |
| All | 0.717 | 0.618 | 0.684 | 0.578 | 0.682 |

The last three columns are the shares of frauds of each scenario flagged at the same
false-positive rate. The standard deviation of AUPRC over seeds is at most 0.003. In the
first fold, trained on 22 days only, AUPRC with all features is 0.555 against 0.76 to
0.79 in the following folds.

### Interpretation

- Terminal aggregates are necessary and sufficient to detect frauds on compromised
  terminals, which no other feature reveals (0.001 without them).
- Customer aggregates raise the detection of frauds on compromised customers from 0.32
  to 0.82, and that of large amounts to 0.98.
- Calendar flags bring nothing; in the simulator, fraud does not depend on the day of
  the week nor on the hour.
- With all groups, AUPRC is highest but the first and third scenarios are less well
  detected at a 0.1% false-positive rate than with customer aggregates alone. A
  plausible explanation, not verified here, is that legitimate transactions on terminals
  whose compromise has just ended keep a high delayed risk and consume part of the
  false-positive budget.
- With raw features, only 54% of the transactions above the amount threshold are
  flagged, although a single rule on the amount would flag all of them. Inspection of
  one fold shows that the model assigns them scores near 0.55, below those of some
  legitimate transactions close to the threshold. The configuration retained in D2 draws
  about 62% of the features for each tree, that is a single feature when only three are
  available, and is strongly regularised: it under-fits the smallest feature sets.

### Caveats

- The data are synthetic and the scenarios were designed so that these features are
  informative. The result validates the feature pipeline; it says nothing about the
  size of the benefit on real transactions.
- Hyper-parameters were not tuned on the simulated data. The comparison is therefore
  unfavourable to the small feature sets, and the absolute values are lower bounds.
- The seven-day label delay makes the first week of each terminal compromise
  undetectable by the terminal risk, which bounds the recall on that scenario.

### Next

Step 4: cost-based decision threshold on the main dataset.
