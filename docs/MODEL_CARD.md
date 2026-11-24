# Model card — fraud detection model, version 0.2.0

The tables of this document are generated from `results/` by `scripts/update_readme.py`.

## Model

| | |
|---|---|
| Task | Binary classification of card transactions: fraud or legitimate |
| Output | Calibrated probability of fraud; alert decision; contribution of each feature |
| Classifier | Gradient-boosted trees (XGBoost), 500 trees of depth 4; hyper-parameters in `configs/config.yaml` |
| Calibration | Sigmoid (Platt scaling), fitted on data withheld from the classifier |
| Decision rule | Alert when probability × amount exceeds the handling cost of an alert (10 by default) |
| Explanation | Shapley contributions (TreeSHAP), additive on the calibrated log-odds scale |
| Files | `models/fraud_model/model.json` (XGBoost format) and `metadata.json`; no pickled object |
| Provenance | Data checksum and library versions are recorded in `metadata.json`; the commit recorded is the one preceding the commit that added the training script |

## Intended use

Demonstration of a fraud detection pipeline and of its evaluation. The model can be
queried through the API of this repository to study its behaviour on transactions with
the schema of its training data.

## Out of scope

- Decisions on real transactions. The model was trained on two days of transactions
  made in 2013 and described by anonymised features that only the provider of the
  dataset can compute.
- Any use as a production service: the API has no authentication, rate limiting,
  logging or monitoring.
- Transactions from another period, market or card scheme: no evidence is available on
  how the model transfers.

## Training data

Credit Card Fraud Detection dataset (Worldline and the Machine Learning Group of ULB):
284,807 transactions over two days of September 2013, 492 frauds, 28 principal
components, amount and elapsed time. Exact duplicate rows are removed. The model is
trained on the first 80% of the transactions in chronological order: the classifier on
181,585 transactions (347 frauds), the calibrator on the following 45,396 (52 frauds).

## Evaluation

The model was selected by expanding-window cross-validation on its training period and
evaluated once on the last 20% of the transactions (56,745 transactions, 74 frauds,
covering 7.7 hours). It is compared with the model of the first version of the project
on the same transactions; that first version had been evaluated on them once before,
at the start of the revision. Intervals are 95% bootstrap intervals over transactions;
each system was trained with a single seed.

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

Savings are the share of the fraudulent amount that is avoided, net of the cost of the
alerts, under the cost model of the decision rule.

## Limitations

- **Statistical power.** The evaluation set holds 74 frauds. The intervals are wide and
  the improvement over the first version is not established: the intervals of the
  differences include zero.
- **Time span.** Two days of data: the model has not been exposed to weekly or seasonal
  patterns, nor to changes in fraud behaviour. Performance varied substantially between
  consecutive periods in cross-validation.
- **Decision rule.** The rule ignores frauds whose amount is below the alert cost
  divided by their probability; on the evaluation set it flags about three frauds in
  ten, and fewer than three of its alerts in ten are frauds. Its advantage over a
  probability threshold of 0.5, estimated in cross-validation, was not reproduced on
  the evaluation set at alert costs of 5 and 10. It values each transaction in
  isolation and relies on assumed costs.
- **Calibration.** In cross-validation the probabilities are slightly over-confident
  above 0.001, most clearly between 0.2 and 0.5, the range in which the rule decides.
- **Features.** No customer or merchant history is available in this dataset.
  Behavioural features were studied on simulated data only.
- **Explanations.** They name anonymised components and describe the model, not the
  causes of fraud. Only the four leading features have a stable rank across periods.

## Ethical considerations

- A false alert may block or delay a legitimate payment. The cost model reduces this
  harm to a fixed handling cost, which understates its effect on the customer.
- The dataset contains no demographic attribute, so the model cannot be audited for
  differences in error rates between groups of customers. Such an audit would be
  required before any real use.
- Under the retained rule, frauds of small amount are knowingly left undetected. This
  is acceptable only where the loss is borne by the institution and not by the
  cardholder.

## Maintenance

The model is static. A deployed version would require monitoring of the distribution of
its inputs and of its alert rate, periodic re-training on recent labelled transactions,
and re-calibration.
