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
