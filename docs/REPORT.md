# Credit card fraud detection: from an optimistic baseline to an evaluated prototype

Technical report, version 2 of the project. TAMPLARE K. Marcel, November 2026.

The tables of this report are generated from `results/` by `scripts/update_readme.py`.
Detailed results are in the [README](../README.md), day-to-day observations in the
[journal](JOURNAL.md), and the reasons for each choice in the [decision log](DECISIONS.md).

## Summary

The first version of this project trained a random forest on the Credit Card Fraud
Detection dataset and reported a precision of 0.94 and a recall of 0.82. This report
describes its revision. The original evaluation is shown to be optimistic: its random
split placed copies of test transactions in the training set and ignored chronology. A
protocol based on a temporal hold-out set and expanding-window cross-validation is put
in place, under which model families, imbalance treatments, calibration methods and
decision rules are compared. The retained system, gradient-boosted trees with sigmoid
calibration and a cost-based decision rule, is served by a validated API with exact
explanations. On the hold-out set it improves on the first version in ranking quality
and in cost, but the number of frauds available does not allow the improvement to be
established at the 95% level.

## 1. Problem and data

The task is to flag fraudulent card transactions among 284,807 transactions made over
two days, of which 492 (0.17%) are frauds. The features are 28 anonymised principal
components, the amount and the elapsed time. Two properties of the data constrain every
conclusion: the frauds are few, and the period is short.

## 2. Evaluation protocol

Exact duplicate rows (1,081, of which 19 frauds) are removed. The last 20% of the
transactions in chronological order form a hold-out set (74 frauds), evaluated once at
the end. Every choice is made on the first 80% by expanding-window cross-validation
over four periods. The primary metric is the area under the precision-recall curve
(AUPRC); the recall at a false-positive rate of at most 0.1% and a cost-based measure
complement it. Unit tests enforce the absence of leakage between partitions.

Under the protocol of the first version, 344 test rows were copies of training rows.
The same model evaluated under the temporal protocol loses recall and AUPRC:

<!-- BEGIN:protocol -->
| Evaluation protocol | Frauds in test | Precision | Recall | F1 | AUPRC [95% CI] |
|---|---|---|---|---|---|
| Random stratified split, duplicates kept (first version) | 98 | 0.941 | 0.816 | 0.874 | 0.873 [0.808, 0.934] |
| Temporal hold-out, duplicates removed | 74 | 0.962 | 0.689 | 0.803 | 0.784 [0.689, 0.869] |
<!-- END:protocol -->

## 3. Model selection

<!-- BEGIN:models -->
| Model | AUPRC, mean over seeds (std) | AUPRC per validation period | Recall at FPR ≤ 0.1% |
|---|---|---|---|
| Random forest (first version) | 0.757 (0.007) | 0.804, 0.652, 0.811, 0.762 | 0.803 |
| Logistic regression | 0.749 (0.000) | 0.741, 0.628, 0.859, 0.768 | 0.792 |
| LightGBM | 0.778 (0.003) | 0.814, 0.697, 0.817, 0.786 | 0.803 |
| XGBoost | 0.780 (0.002) | 0.812, 0.703, 0.820, 0.786 | 0.795 |
<!-- END:models -->

Gradient-boosted trees exceed the random forest in every validation period and for
every seed, by a small margin. LightGBM and XGBoost are not distinguishable in
accuracy; XGBoost is retained for its lower sensitivity to settings. No treatment of the
class imbalance (class weighting, under-sampling, SMOTE) improves on the untreated
model. Sigmoid calibration improves the probabilities without altering the ranking.

## 4. Decision rule

A probability is turned into a decision under an explicit cost model: an alert has a
fixed handling cost and a missed fraud costs its amount. The rule that follows from this
model alerts when probability × amount exceeds the alert cost.

<!-- BEGIN:costs -->
| Decision rule | Savings, alert cost 5 | Savings, alert cost 10 | Savings, alert cost 25 | Alerts per 100,000 (alert cost 10) | Precision | Recall |
|---|---|---|---|---|---|---|
| Probability threshold of 0.5 (first version) | 66.0% | 63.1% | 54.6% | 116 | 0.874 | 0.716 |
| Cost-optimal probability threshold | 66.0% | 62.3% | 53.0% | 140 | 0.749 | 0.739 |
| Expected-cost rule (probability x amount > alert cost) | 71.0% | 66.3% | 60.6% | 134 | 0.435 | 0.412 |
| Oracle (upper bound) | 97.6% | 95.7% | 90.4% | | | |
<!-- END:costs -->

In cross-validation this rule gives the highest savings. A single cost-tuned threshold
cannot be estimated reliably with the number of frauds available.

## 5. Behavioural features

The main dataset has no identifier. Customer and terminal aggregates were therefore
implemented and tested on simulated transactions following the specification of the
*Fraud Detection Handbook*, with a label delay of seven days and tests of point-in-time
correctness.

<!-- BEGIN:behavioural -->
| Feature set | Features | AUPRC, mean over seeds (std) | Recall at FPR ≤ 0.1% | Recall, amount above threshold | Recall, compromised terminal | Recall, compromised customer |
|---|---|---|---|---|---|---|
| Amount and time of day | 3 | 0.164 (0.000) | 0.139 | 0.541 | 0.001 | 0.324 |
| + calendar flags | 5 | 0.163 (0.000) | 0.139 | 0.545 | 0.001 | 0.324 |
| + customer aggregates | 12 | 0.349 (0.000) | 0.323 | 0.976 | 0.002 | 0.819 |
| + terminal aggregates | 9 | 0.590 (0.001) | 0.419 | 0.275 | 0.559 | 0.171 |
| All groups | 20 | 0.717 (0.003) | 0.618 | 0.684 | 0.578 | 0.682 |
<!-- END:behavioural -->

The scenarios of the simulator were designed so that such features are informative:
this result validates the feature pipeline, not a gain on real transactions.

## 6. Final evaluation

The stored model and the first version, trained on the same period, are compared on the
hold-out set.

<!-- BEGIN:final -->
| System | AUPRC | Recall at FPR ≤ 0.1% | Savings, alert cost 10 | Alerts | Frauds flagged |
|---|---|---|---|---|---|
| First version: random forest, threshold 0.5 | 0.784 [0.689, 0.869] | 0.797 [0.689, 0.883] | 42.4% [13.3%, 71.2%] | 53 | 68.9% |
| Retained model, threshold 0.5 | 0.803 [0.709, 0.883] | 0.797 [0.696, 0.887] | 57.8% [29.2%, 83.7%] | 57 | 73.0% |
| Retained model, expected-cost rule | 0.803 [0.709, 0.883] | 0.797 [0.696, 0.887] | 58.2% [28.8%, 82.5%] | 77 | 29.7% |
| Difference, last row minus first | +0.019 [-0.004, +0.046] | +0.000 [-0.038, +0.044] | +15.9% [-3.8%, +41.7%] | | |
| Share of bootstrap resamples in which the last row is better | 94% | 38% | 90% | | |
<!-- END:final -->

<!-- BEGIN:final_sensitivity -->
| System | Savings, alert cost 5 | Savings, alert cost 10 | Savings, alert cost 25 |
|---|---|---|---|
| First version: random forest, threshold 0.5 | 45.8% | 42.4% | 32.1% |
| Retained model, threshold 0.5 | 61.5% | 57.8% | 46.8% |
| Retained model, expected-cost rule | 60.5% | 58.2% | 52.5% |
| Oracle (upper bound) | 97.1% | 95.0% | 89.7% |
<!-- END:final_sensitivity -->

## 7. Discussion

**Established.** The evaluation of the first version was biased upwards, by a mechanism
that is identified and measured (duplicates across the split, absence of chronology).
The pipeline is reproducible: every figure is produced by a script, stored with its
provenance, and the leakage guards run in continuous integration.

**Supported but not established.** The retained model ranks transactions better than
the first version and reduces the cost of fraud; both effects have the expected sign,
agree with cross-validation, and hold in most bootstrap resamples, but their 95%
intervals include zero.

**Not confirmed.** The advantage of the expected-cost rule over a threshold of 0.5,
observed in cross-validation, does not appear on the hold-out set for alert costs of 5
and 10. On that set, the gain in savings comes from the model.

**Negative results.** Imbalance treatments, calendar features and the replacement of the
elapsed time by the time of day bring no measurable gain. The recall at a low
false-positive rate is not improved by any model.

## 8. Limitations and further work

The conclusions are bounded by the 74 frauds of the hold-out set and by the two days of
data. A dataset covering several months, with identifiers, would allow behavioural
features to be evaluated on real transactions, drift to be studied, and differences
between models to be resolved. The cost model values transactions in isolation and
rests on assumed costs. The service is a prototype without authentication or
monitoring. Deep tabular models and ensembles of the retained models were not explored.
