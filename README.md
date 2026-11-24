# Credit Card Fraud Detection

Supervised detection of fraudulent card transactions on a highly imbalanced dataset
(0.17% of frauds), evaluated under a protocol that respects the chronology of the
transactions, with the aim of a deployable prototype: a trained and calibrated model,
a validated REST API and an interactive dashboard.

> **Status — version 2 in progress.** Steps 1 and 2 of the roadmap below are complete:
> evaluation protocol, reference measurements and model selection. The serving layer
> (`api/`, `dashboard/`) still dates from the first version and is rewritten in step 6.
> The first version is preserved under the tag `v0-legacy`.

## Contents

1. [Results so far](#results-so-far)
2. [Method](#method)
3. [Reproducing the results](#reproducing-the-results)
4. [Repository layout](#repository-layout)
5. [Roadmap](#roadmap)
6. [Limitations](#limitations)
7. [History](#history)
8. [References](#references)

## Results so far

All tables below are generated from the files in `results/` by
`scripts/update_readme.py`. The reasoning behind each choice is recorded in
[`docs/DECISIONS.md`](docs/DECISIONS.md) and the day-to-day observations in
[`docs/JOURNAL.md`](docs/JOURNAL.md).

### 1. The evaluation protocol of the first version was optimistic

The same model (random forest, 100 trees, default settings) is evaluated under the
protocol of the first version and under a protocol that removes duplicated rows and
tests on transactions strictly later than the training period.

<!-- BEGIN:protocol -->
| Evaluation protocol | Frauds in test | Precision | Recall | F1 | AUPRC [95% CI] |
|---|---|---|---|---|---|
| Random stratified split, duplicates kept (first version) | 98 | 0.941 | 0.816 | 0.874 | 0.873 [0.808, 0.934] |
| Temporal hold-out, duplicates removed | 74 | 0.962 | 0.689 | 0.803 | 0.784 [0.689, 0.869] |
<!-- END:protocol -->

Precision, recall and F1 are computed at a threshold of 0.5. Under the first protocol,
344 test rows, 8 of them frauds, were exact copies of training rows, and the random
split let the model train on transactions later than those it was tested on. The two
test sets differ and their confidence intervals overlap: the comparison shows the
direction of the bias rather than its exact size.

### 2. Gradient-boosted trees improve on the random forest by a small, consistent margin

Expanding-window cross-validation over four consecutive periods of the development set,
five seeds, no imbalance treatment. The hold-out set is not used.

<!-- BEGIN:models -->
| Model | AUPRC, mean over seeds (std) | AUPRC per validation period | Recall at FPR ≤ 0.1% |
|---|---|---|---|
| Random forest (first version) | 0.757 (0.007) | 0.804, 0.652, 0.811, 0.762 | 0.803 |
| Logistic regression | 0.749 (0.000) | 0.741, 0.628, 0.859, 0.768 | 0.792 |
| LightGBM | 0.778 (0.003) | 0.814, 0.697, 0.817, 0.786 | 0.803 |
| XGBoost | 0.780 (0.002) | 0.812, 0.703, 0.820, 0.786 | 0.795 |
<!-- END:models -->

Both boosted models exceed the random forest on each of the four periods and for each of
the five seeds. The two boosted models cannot be told apart on AUPRC, and no model
improves the recall at a low false-positive rate. Performance differs far more between
periods than between models, which bounds what this dataset can resolve.

XGBoost is retained. With equal accuracy, it proved the more robust of the two to its
settings during the search (see decision D2).

### 3. Correcting the class imbalance does not help

<!-- BEGIN:imbalance -->
| Imbalance treatment | AUPRC, mean over seeds (std) |
|---|---|
| None | 0.780 (0.002) |
| Class weighting | 0.775 (0.001) |
| Random under-sampling | 0.747 (0.008) |
| SMOTE | 0.780 (0.002) |
<!-- END:imbalance -->

Treatments are applied to the training folds only. None of them improves on the
untreated model; under-sampling discards information and lowers performance. The model
is therefore trained on the original class proportions.

### 4. Sigmoid calibration improves the predicted probabilities

<!-- BEGIN:calibration -->
| Calibration | Brier score | Log loss | AUPRC |
|---|---|---|---|
| None | 0.000522 | 0.00314 | 0.772 |
| Sigmoid (Platt) | 0.000483 | 0.00312 | 0.772 |
| Isotonic | 0.000554 | 0.00637 | 0.726 |
<!-- END:calibration -->

The classifier is fitted on the first part of each training window and the calibrator on
the last part. Sigmoid calibration lowers the Brier score without altering the ranking
of the transactions. Isotonic calibration, fitted on a few tens of frauds, introduces
ties that degrade the ranking.

### Not yet available

The performance of the retained model on the hold-out set is deliberately not reported
at this stage. It is measured once, at the end, so that no modelling choice can be
influenced by it. The cross-validated scores above are those of the best configuration
among those searched and are therefore slightly optimistic.

## Method

**Data.** [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
dataset (Worldline and the Machine Learning Group of ULB): 284,807 transactions made
over two days in September 2013, 492 of them frauds. The features are 28 principal
components (`V1`–`V28`), the amount and the time elapsed since the first transaction.
The file is not distributed with this repository; its checksum is verified before use.

**Preparation.** Rows that are exact copies of an earlier row are removed (1,081 rows,
19 frauds). The elapsed time, which has no defined value outside the collection period,
is replaced by a cyclic encoding of the time of day.

**Split.** The last 20% of the transactions in chronological order form the hold-out set
(74 frauds). The first 80% form the development set (399 frauds), on which every choice
is made by expanding-window cross-validation: each fold trains on the past and validates
on the period that follows.

**Metrics.** The primary metric is the area under the precision-recall curve (AUPRC),
which does not depend on a threshold and is suited to rare positives. The recall at a
false-positive rate of at most 0.1% is reported as a secondary metric. Confidence
intervals are obtained by bootstrap.

**Safeguards.** Unit tests check that no partition shares a row with another, that no
model is validated on transactions older than its training data, and that re-sampling is
never applied to validation data. They run in continuous integration on every push.

## Reproducing the results

Python 3.11 is required.

```bash
git clone https://github.com/MarcelKossi/Fraud_detection_system.git
cd Fraud_detection_system
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Download `creditcard.csv` from the dataset page and place it in `data/raw/`. Then:

```bash
pytest -q                               # unit tests and leakage guards
python scripts/prepare_data.py          # checksum, duplicates, partitions
python scripts/run_baseline.py          # result 1, about 10 minutes on 2 cores
python scripts/run_model_selection.py   # results 2 to 4, about 35 minutes on 2 cores
python scripts/update_readme.py         # regenerate the tables of this file
```

Every experiment reads `configs/config.yaml` and writes a JSON file to `results/`
containing its metrics together with the commit, the data checksum, the configuration
and the library versions.

## Repository layout

```
configs/config.yaml        data, split, evaluation and model settings
src/fraud_detection/       installable package
    data.py                loading, checksum, de-duplication
    features.py            feature sets
    splits.py              temporal hold-out, expanding-window folds, leakage checks
    metrics.py             AUPRC, recall at fixed FPR, bootstrap intervals
    models.py              model definitions
    sampling.py            imbalance treatments
    search.py              hyper-parameter search spaces
    calibration.py         probability calibration
    validation.py          cross-validated evaluation
    experiment.py          provenance of results
scripts/                   experiment entry points
tests/                     unit tests
results/                   metrics (JSON) and generated summaries
docs/                      journal and decision log
api/, dashboard/           serving layer of the first version (rewritten in step 6)
notebooks/                 exploratory analysis of the first version
```

## Roadmap

| Step | Content | State |
|---|---|---|
| 1 | Reference measurements, temporal protocol, tests | Done |
| 2 | Model selection, imbalance treatments, calibration | Done |
| 3 | Behavioural features on a dataset with card and terminal identifiers | Planned |
| 4 | Cost-based decision threshold | Planned |
| 5 | Explanation of individual predictions | Planned |
| 6 | Deployable prototype: single pipeline, validated API, container | Planned |
| 7 | Final evaluation on the hold-out set, model card, technical report | Planned |

## Limitations

- The dataset covers two days. Long-term changes in fraud patterns cannot be studied,
  and the hold-out set contains 74 frauds only: confidence intervals are wide.
- The features are anonymised principal components. They cannot be interpreted in
  business terms and do not allow aggregates per card or per merchant.
- Differences between models below about 0.01 of AUPRC cannot be resolved with the
  number of frauds available.
- The hyper-parameter search is limited to 30 random configurations per boosted model.

## History

The first version of this project trained a random forest with default settings and
reported precision, recall and F1 on a random split. Those figures are reproduced
exactly by `scripts/run_baseline.py`. Version 2 was started after finding that this
split places identical transactions on both sides and ignores chronology. The code of
the first version remains available under the tag `v0-legacy`.

## References

- A. Dal Pozzolo, O. Caelen, R. A. Johnson and G. Bontempi. *Calibrating Probability
  with Undersampling for Unbalanced Classification.* IEEE Symposium Series on
  Computational Intelligence, 2015. (Source of the dataset.)
- J. Davis and M. Goadrich. *The Relationship Between Precision-Recall and ROC Curves.*
  International Conference on Machine Learning, 2006.
- N. V. Chawla, K. W. Bowyer, L. O. Hall and W. P. Kegelmeyer. *SMOTE: Synthetic
  Minority Over-sampling Technique.* Journal of Artificial Intelligence Research, 2002.
- J. Platt. *Probabilistic Outputs for Support Vector Machines and Comparisons to
  Regularized Likelihood Methods.* Advances in Large Margin Classifiers, 1999.
- T. Chen and C. Guestrin. *XGBoost: A Scalable Tree Boosting System.* ACM SIGKDD, 2016.
- G. Ke et al. *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* Advances
  in Neural Information Processing Systems, 2017.

## Author

TAMPLARE K. Marcel

## Purpose and terms of use

This project is a demonstration of technical expertise in machine learning, conducted
with the methodological standards of research. It is shared for demonstration and
research purposes.
