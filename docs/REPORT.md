# Credit card fraud detection: from an optimistic baseline to an evaluated prototype

Technical report, version 2 of the project. Tamplare Kossi, November 2026.

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
calibration and a cost-based decision rule, is served by an API with input validation
and additive explanations. On the hold-out set its point estimates of ranking quality
and of savings are higher than those of the first version, but the intervals of the
differences include zero: with the number of frauds available, the improvement is not
established.

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
The same model has a lower recall and AUPRC on the temporal hold-out set, with
overlapping intervals:

<!-- BEGIN:protocol -->
| Evaluation protocol | Frauds in test | Precision | Recall | F1 | AUPRC [95% CI] |
|---|---|---|---|---|---|
| Random stratified split, duplicates kept (first version) | 98 | 0.941 | 0.816 | 0.874 | 0.873 [0.808, 0.934] |
| Temporal hold-out, duplicates removed | 74 | 0.962 | 0.689 | 0.803 | 0.784 [0.689, 0.869] |
<!-- END:protocol -->

Repeating the random split with and without the duplicated rows separates the two
defects:

<!-- BEGIN:decomposition -->
| Protocol | AUPRC, mean (std over splits) | Recall at 0.5, mean (std) | Test frauds also in training, mean |
|---|---|---|---|
| Random splits, duplicates kept (protocol of the first version) | 0.855 (0.032) | 0.794 (0.045) | 5.8 |
| Random splits, duplicates removed | 0.817 (0.034) | 0.771 (0.034) | 0.0 |
| Temporal folds, duplicates removed | 0.752 (0.073) | 0.692 (0.108) | 0 |
<!-- END:decomposition -->

Removing duplicates lowers the scores of the random splits by an amount comparable to
their variation between splits. The gap to the temporal folds also includes the effect
of their smaller training sets and of their particular periods.

## 3. Model selection

<!-- BEGIN:models -->
| Model | AUPRC, mean over seeds (seed std) | AUPRC per validation period | Recall at FPR ≤ 0.1% |
|---|---|---|---|
| Random forest, default settings | 0.757 (0.007) | 0.804, 0.652, 0.811, 0.762 | 0.803 |
| Logistic regression | 0.749 (0.000) | 0.741, 0.628, 0.859, 0.768 | 0.792 |
| LightGBM | 0.778 (0.003) | 0.814, 0.697, 0.817, 0.786 | 0.803 |
| XGBoost | 0.780 (0.002) | 0.812, 0.703, 0.820, 0.786 | 0.795 |
<!-- END:models -->

Gradient-boosted trees exceed the untuned random forest on the seed-averaged AUPRC of
every validation period, by a small margin whose uncertainty is given below. LightGBM
and XGBoost are not distinguishable in accuracy; XGBoost is retained. Under the
settings tested, no treatment of the class imbalance (class weighting, under-sampling,
SMOTE) improves AUPRC. Sigmoid calibration lowers the Brier score without altering the
ranking; the calibrated probabilities remain slightly over-confident.

<!-- BEGIN:model_comparison -->
| Quantity | Value |
|---|---|
| Mean AUPRC, retained model | 0.780 |
| Mean AUPRC, random forest | 0.757 |
| Difference, with 95% paired bootstrap interval | +0.023 [+0.004, +0.041] |
| Share of resamples in which the difference is positive | 99% |
| Difference in each validation period | +0.008, +0.051, +0.009, +0.024 |
| Pairs of seeds won by the retained model, in each period | 25 of 25, 25 of 25, 16 of 25, 25 of 25 |
<!-- END:model_comparison -->

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

In cross-validation this rule gives the highest point estimate of savings. Its
advantage over the threshold of 0.5 is established for an alert cost of 25 only:

<!-- BEGIN:cost_intervals -->
| Alert cost | Savings of the expected-cost rule minus savings of the threshold of 0.5 | 95% paired bootstrap interval | Share of resamples above zero |
|---|---|---|---|
| 5 | +5.0 points | [+0.2, +10.8] | 98% |
| 10 | +3.2 points | [-0.5, +7.8] | 95% |
| 25 | +6.0 points | [+3.0, +10.2] | 100% |
<!-- END:cost_intervals -->

A single cost-tuned threshold cannot be estimated reliably with the number of frauds
available.

## 5. Behavioural features

The main dataset has no identifier. Customer and terminal aggregates were therefore
implemented and tested on simulated transactions following the specification of the
*Fraud Detection Handbook*, with a label delay of seven days and tests of point-in-time
correctness.

<!-- BEGIN:behavioural -->
| Feature set | Features | AUPRC, mean over seeds (seed std) | Recall at FPR ≤ 0.1% | Recall, amount above threshold | Recall, compromised terminal | Recall, compromised customer |
|---|---|---|---|---|---|---|
| Amount and time of day | 3 | 0.164 (0.000) | 0.139 | 0.541 | 0.001 | 0.324 |
| Amount, time of day and calendar flags | 5 | 0.163 (0.000) | 0.139 | 0.545 | 0.001 | 0.324 |
| Amount, time of day and customer aggregates | 12 | 0.349 (0.000) | 0.323 | 0.976 | 0.002 | 0.819 |
| Amount, time of day and terminal aggregates | 9 | 0.590 (0.001) | 0.419 | 0.275 | 0.559 | 0.171 |
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
| Share of bootstrap resamples in which the last row is strictly better | 94% | 38% (ties counted as not better) | 90% | | |
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

**Established.** The protocol of the first version placed copies of test transactions
in its training set and did not respect chronology; both facts are verified on the
data. In cross-validation, the tuned boosted model has a higher AUPRC than the untuned
random forest (paired bootstrap interval excluding zero). The experiments are
deterministic and their tables are generated from
stored results; the leakage guards run in continuous integration.

**Not established.** The size of the bias of the first version: removing duplicates
and respecting chronology both lower its scores, but the first effect is of the order
of the variation between random splits, and the second is confounded with training-set
size and with the period tested. That the retained model
ranks transactions better than the first version and reduces the cost of fraud: the
point estimates have the expected sign and agree with cross-validation, but their 95%
intervals include zero, and each system was trained with a single seed.

**Not confirmed.** The advantage of the expected-cost rule over a threshold of 0.5 does
not appear on the hold-out set for alert costs of 5 and 10, as the cross-validation
intervals allowed. On that set, the difference in savings comes from the model.

**Negative results.** Imbalance treatments, calendar features and the replacement of the
elapsed time by the time of day bring no measurable gain. The recall at a low
false-positive rate is not improved by any model.

## 8. Limitations and further work

The conclusions are bounded by the 74 frauds of the hold-out set and by the two days of
data. A dataset covering several months, with identifiers, would allow behavioural
features to be evaluated on real transactions, drift to be studied, and differences
between models to be resolved. The cost model values transactions in isolation and
rests on assumed costs. The bootstrap intervals treat transactions as independent and
do not include the variability of training. The result files produced before the
review of the project record the commit that precedes their generating script. The
service is a prototype without authentication or monitoring. Deep tabular models and ensembles of the retained models were not explored.
