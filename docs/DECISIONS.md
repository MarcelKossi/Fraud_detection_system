# Decision log

Each entry states the question, the options considered, the evidence, and the choice.

---

## D1 — Data, evaluation protocol and primary metric (2026-11-24)

### Datasets

**Choice.** The Credit Card Fraud Detection dataset (ULB / Worldline) remains the main
dataset. A second, simulated dataset with card and terminal identifiers (generator of
the *Fraud Detection Handbook*) is added for behavioural feature engineering.

**Reason.** The main dataset is anonymised by PCA and carries no card or merchant
identifier; aggregates per card or per terminal cannot be computed on it.

### Duplicates

**Choice.** Rows that are exact copies of an earlier row are removed before any split.

**Evidence.** 1,081 such rows exist (19 frauds). Under the random split of the first
version, 344 test rows, including 8 frauds, also appear in the training set.

### Split

**Options.** (a) Random stratified split, as in the first version. (b) Fixed temporal
60/20/20 split. (c) Temporal hold-out of the last 20%, with expanding-window
cross-validation on the first 80%.

**Choice.** (c). The random split is kept only to replicate the first version.

**Evidence.** A fraud detector is applied to transactions that follow its training
data; a random split does not reproduce that situation. Option (b) leaves 57 frauds in
the validation set, too few to select a model or a threshold reliably. Option (c) uses
the 399 frauds of the development period across four validation folds (69, 47, 89 and
52 frauds) and keeps a hold-out set of 74 frauds.

**Known limitation.** The dataset spans two days. With 74 frauds in the hold-out set,
confidence intervals remain wide whatever the protocol.

### Time feature

**Choice.** The elapsed time since the first transaction is replaced by a cyclic
encoding of the time of day.

**Evidence.** The elapsed time has no defined value for a transaction recorded after
the collection period, hence cannot be used by a deployed model. The substitution
leaves cross-validated AUPRC unchanged (0.751 against 0.752).

### Primary metric

**Choice.** Area under the precision-recall curve (AUPRC). Secondary metrics: recall at
a false-positive rate of at most 0.1%, and a cost-based measure defined in a later
decision. Precision, recall and F1 at a fixed threshold are reported for continuity
with the first version only.

**Reason.** With 0.17% of positive examples, threshold-dependent metrics at an
arbitrary threshold and metrics dominated by true negatives are uninformative; AUPRC is
threshold-free and is the measure recommended by the authors of the dataset.

---

## D2 — Model, imbalance treatment and calibration (2026-11-24)

All evidence comes from expanding-window cross-validation on the development period,
repeated over five seeds (`results/model_selection/`). The hold-out set was not used.

### Model family

**Options.** Random forest (first version), logistic regression, LightGBM, XGBoost.

**Choice.** XGBoost, with the best configuration of the random search
(`configs/config.yaml`, section `selected_model`).

**Evidence.** Mean AUPRC: random forest 0.757, logistic regression 0.749, LightGBM
0.778, XGBoost 0.780. Both boosted models exceed the random forest on the seed-averaged
AUPRC of every validation period and on the period-averaged AUPRC of every seed (see
the notes of 2026-11-25 at the end of this document). The difference between XGBoost and LightGBM (0.002) is within
seed-to-seed variability: accuracy does not separate them. XGBoost is preferred for its
lower sensitivity to settings: over 30 searched configurations its AUPRC stays between
0.686 and 0.783, against 0.588 to 0.781 for LightGBM, and it remains stable under class
weighting (0.775) where LightGBM falls to 0.616.

**What this choice does not claim.** XGBoost is not shown to be more accurate than
LightGBM. On the secondary metric, recall at a false-positive rate of at most 0.1%, it
is slightly below the random forest and LightGBM (0.795 against 0.803).

### Imbalance treatment

**Options.** None, class weighting, random under-sampling, SMOTE; the last two applied
to training folds only.

**Choice.** None.

**Evidence.** For XGBoost: none 0.780, SMOTE 0.780, class weighting 0.775,
under-sampling 0.747. No treatment exceeds the untreated model by more than twice the
standard deviation over seeds, the criterion set for adopting one. The simplest option
is retained; it also leaves the class proportions, and therefore the predicted
probabilities, undistorted.

### Probability calibration

**Options.** None, sigmoid (Platt scaling), isotonic regression.

**Choice.** Sigmoid, fitted on the most recent part of the training data, which is
withheld from the classifier.

**Evidence.** For XGBoost, the Brier score falls from 0.000522 to 0.000483 with the
sigmoid and AUPRC is unchanged (0.772). Isotonic regression raises the Brier score to
0.000554 and lowers AUPRC to 0.726. Calibrated probabilities are required by the
cost-based threshold of step 4.

---

## D3 — Cost model and decision rule (2026-11-24)

Evidence: `results/cost/`, calibrated out-of-fold probabilities of the retained model on
the four validation periods of the development set, five seeds. The hold-out set was not
used.

### Cost model

**Choice.** Every alert costs a fixed handling cost, whether justified or not; a fraud
that raises no alert costs the amount of the transaction. The reference alert cost is
10, with 5 and 25 as sensitivity values.

**Status of these values.** They are assumptions made for the purpose of the
demonstration, not costs measured at a card issuer. The alert cost is a parameter of the
configuration.

### Decision rule

**Options.** (a) Probability threshold of 0.5, as in the first version. (b) Single
probability threshold minimising the cost. (c) Expected-cost rule: alert when
probability x amount exceeds the alert cost.

**Choice.** (c).

**Evidence.** Savings for alert costs of 5, 10 and 25: (a) 66.0%, 63.1%, 54.6%;
(b) 66.0%, 62.3%, 53.0%; (c) 71.0%, 66.3%, 60.6%. Rule (c) is ahead in every validation
period for alert costs of 10 and 25 and in three of four for 5. Rule (b) cannot be
estimated reliably: its threshold ranges from 0.05 to 0.82 across periods for an alert
cost of 10. Rule (c) follows directly from the cost model and has no fitted parameter.

**Accepted consequence.** Rule (c) flags 41% of the frauds against 72% for rule (a), at
a comparable number of alerts: it gives up frauds of small amount, which are numerous
in this dataset (median fraudulent amount 19), in order to catch the expensive ones.
The objective retained is the reduction of financial loss, not the number of frauds
detected.

**Limits.** Each transaction is valued in isolation; a small fraud that precedes larger
ones on the same card is not accounted for. The rule relies on the calibration of the
probabilities.

---

## Notes added after the review of 2026-11-25

These notes qualify the evidence quoted above. The decisions themselves are unchanged.

**D2, model.** The statement that both boosted models exceed the random forest "on
every validation period and for every seed" was inaccurate: it holds for averages over
seeds and over periods, not for every pair of seeds within a period. The preference for
XGBoost over LightGBM rests on the worst of 30 searched configurations only (0.686
against 0.588); their medians are equal (0.767). The choice is one between equals. The
uncertainty of the gain over the random forest is given by the paired bootstrap of
`results/robustness/model_comparison.json`. The random forest was not tuned.

**D2, imbalance.** The treatments were compared with the hyper-parameters selected
without treatment and with a single re-sampling ratio. Class weighting and SMOTE give a
slightly higher recall at a false-positive rate of 0.1% (0.804 and 0.802 against
0.795).

**D3.** The advantage of the expected-cost rule was first assessed against the
standard deviation over training seeds, which is not the relevant source of
uncertainty. A paired bootstrap over validation transactions gives, for the difference
in savings with the threshold of 0.5: +5.0 points [+0.2, +10.8] at an alert cost of 5,
+3.2 points [-0.5, +7.8] at 10, and +6.0 points [+3.0, +10.2] at 25. The advantage is
established at 25 only. On the hold-out set the differences are -1.1, +0.4 and +5.7
points, which these intervals allow. The rule is kept because it follows from the cost
model and has no fitted parameter, not because its advantage is demonstrated.
