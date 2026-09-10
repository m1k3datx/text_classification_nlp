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
  accuracy, balanced accuracy, macro F1, and ROC AUC for each model.
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

The train/test split is stratified and seeded. TF-IDF vocabulary fitting and
random undersampling occur inside each cross-validation fold and never use the
holdout set. The default undersampling option addresses the original class
imbalance only in training; use `--no-undersampling` to compare against the
original distribution. Sparse TF-IDF matrices are kept sparse throughout.

The bundled data is a historical, manually labeled news sample and may not
represent current news or production class prevalence. A single random
holdout is not a substitute for temporal validation, and reported metrics are
only produced when the script is run locally. The project does not claim a
production-ready model or guarantee that the labels are free of annotation
noise.

## License

MIT; see [LICENSE](LICENSE).
