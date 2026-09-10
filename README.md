# Economic news relevance classification

This repository contains a small, reproducible text-classification workflow for
predicting whether an article is relevant to the U.S. economy. It uses the
public `US-Economic-News.csv` dataset included in `data/` and compares
logistic regression, a linear support vector classifier, and multinomial naive
Bayes.

## Setup

Use Python 3.10 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

From the repository root:

```powershell
python text_classification.py
```

The default run reads `data/US-Economic-News.csv`, uses a fixed seed (`123`),
holds out 20% of the labeled data, and performs five-fold stratified
cross-validation on the training split. It writes:

- `outputs/model_comparison.csv`: cross-validation mean/std and holdout
  accuracy, balanced accuracy, macro F1, and ROC AUC for each model. The
  selected model is marked with `selected_by_cv`; holdout scores are never used
  for model selection. The `dummy_majority` row is a majority-class baseline.
- `outputs/classification_reports.json`: per-class precision, recall, F1, and
  support for each model.

Useful options:

```powershell
python text_classification.py --data path\to\data.csv --output-dir outputs\run-1
python text_classification.py --seed 7 --test-size 0.25 --cv 3 --jobs -1
python text_classification.py --no-undersampling
```

## Data contract

The input CSV must contain `text` and `relevance` columns. `relevance` must
use `yes` and `no`; `not sure` rows are intentionally excluded because they
are not binary labels. Text is normalized with a lightweight tokenizer and
English stop-word list, so no NLTK downloads are required.

## Methodology and limitations

The normalized text is deduplicated before the stratified train/test split;
rows with the same normalized text but conflicting labels are excluded. This
prevents duplicate articles from crossing the holdout or cross-validation
folds. TF-IDF vocabulary fitting and random undersampling occur inside each
cross-validation fold and never use the holdout set. Models are selected by
cross-validated macro F1 on the training split, then evaluated on the untouched
holdout. The default undersampling option addresses the original class
imbalance only in training; use `--no-undersampling` to compare against the
original distribution. Sparse TF-IDF matrices are kept sparse throughout.

The bundled data is a historical, manually labeled news sample and may not
represent current news or production class prevalence. A single random
holdout is not a substitute for temporal validation, and the metrics in this
README are **preliminary until the workflow is rerun**. Results are sensitive
to normalization, deduplication, class imbalance, and annotation noise; this
is not a production-ready model.

See [data/README.md](data/README.md) for dataset provenance, labeling, and
usage notes. The repository's MIT license applies to the code in this
repository; it does not grant rights to third-party news text or other
dataset contents.

## License

MIT for the repository code; see [LICENSE](LICENSE). Third-party dataset
content remains subject to its own rights and terms.
