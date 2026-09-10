"""Train and compare reproducible text-classification models.

The script expects a CSV with a ``text`` column and a ``relevance`` column
whose values are ``yes``, ``no``, or ``not sure``.  Vectorization and
undersampling are fitted inside each training fold to avoid data leakage.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

SEED = 123
LABEL_NAMES = {0: "not relevant", 1: "relevant"}
TOKEN_PATTERN = re.compile(r"[a-z]+")


def clean_text(text: str) -> str:
    """Normalize article text without requiring a downloaded NLP corpus."""
    text = html.unescape(str(text)).lower().replace("</br>", " ")
    tokens = TOKEN_PATTERN.findall(text)
    return " ".join(token for token in tokens if token not in ENGLISH_STOP_WORDS)


def load_dataset(path: Path) -> pd.DataFrame:
    """Load and validate the expected dataset columns and labels."""
    data = pd.read_csv(path, encoding="ISO-8859-1", usecols=["text", "relevance"])
    data = data.dropna(subset=["text", "relevance"]).copy()
    data["relevance"] = data["relevance"].astype(str).str.lower().str.strip()
    data = data[data["relevance"].isin({"yes", "no"})].copy()
    if data.empty or data["relevance"].nunique() != 2:
        raise ValueError("Dataset must contain both 'yes' and 'no' relevance labels.")
    data["text"] = data["text"].map(clean_text)
    data = data[data["text"].str.len() > 0].copy()
    data["relevance"] = data["relevance"].map({"no": 0, "yes": 1}).astype(int)
    return data[["text", "relevance"]]


def build_models(seed: int, max_features: int, undersample: bool) -> dict[str, Pipeline]:
    """Create pipelines so every learned preprocessing step stays in the fold."""
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    sampler: Any = RandomUnderSampler(random_state=seed) if undersample else "passthrough"
    return {
        "logistic_regression": Pipeline(
            [
                ("tfidf", vectorizer),
                ("sampler", sampler),
                ("classifier", LogisticRegression(max_iter=1000, random_state=seed)),
            ]
        ),
        "linear_svc": Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))),
                ("sampler", sampler),
                ("classifier", LinearSVC(class_weight="balanced", random_state=seed)),
            ]
        ),
        "multinomial_nb": Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))),
                ("sampler", sampler),
                ("classifier", MultinomialNB()),
            ]
        ),
    }


def score_model(model: Pipeline, x_test: pd.Series, y_test: pd.Series) -> tuple[dict[str, float], dict[str, Any]]:
    """Return comparable holdout metrics and a per-class report."""
    predictions = model.predict(x_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "precision_macro": precision_score(y_test, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, predictions, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, predictions, average="macro", zero_division=0),
    }
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(x_test)[:, 1]
    else:
        scores = model.decision_function(x_test)
    metrics["roc_auc"] = roc_auc_score(y_test, scores)
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=[LABEL_NAMES[0], LABEL_NAMES[1]],
        output_dict=True,
        zero_division=0,
    )
    return metrics, report


def run(args: argparse.Namespace) -> pd.DataFrame:
    data = load_dataset(args.data)
    x_train, x_test, y_train, y_test = train_test_split(
        data["text"],
        data["relevance"],
        test_size=args.test_size,
        random_state=args.seed,
        stratify=data["relevance"],
    )
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": "f1_macro",
        "roc_auc": "roc_auc",
    }
    results: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}

    for name, model in build_models(args.seed, args.max_features, args.undersample).items():
        cv_scores = cross_validate(model, x_train, y_train, cv=cv, scoring=scoring, n_jobs=args.jobs)
        model.fit(x_train, y_train)
        test_metrics, report = score_model(model, x_test, y_test)
        results.append(
            {
                "model": name,
                **{f"cv_{metric}_mean": cv_scores[f"test_{metric}"].mean() for metric in scoring},
                **{f"cv_{metric}_std": cv_scores[f"test_{metric}"].std() for metric in scoring},
                **{f"test_{metric}": value for metric, value in test_metrics.items()},
            }
        )
        reports[name] = report

    comparison = pd.DataFrame(results).sort_values("test_f1_macro", ascending=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output_dir / "model_comparison.csv", index=False)
    (args.output_dir / "classification_reports.json").write_text(
        json.dumps(reports, indent=2), encoding="utf-8"
    )

    print(f"Loaded {len(data):,} labeled articles from {args.data}")
    print(f"Train/test split: {len(x_train):,}/{len(x_test):,} (seed={args.seed})")
    print(comparison[["model", "cv_f1_macro_mean", "test_f1_macro", "test_roc_auc"]].to_string(index=False))
    print(f"Reports written to {args.output_dir}")
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "US-Economic-News.csv",
        help="CSV path (default: bundled dataset)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Artifact directory")
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout fraction")
    parser.add_argument("--cv", type=int, default=5, help="Number of stratified CV folds")
    parser.add_argument("--max-features", type=int, default=20000, help="TF-IDF vocabulary limit")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel CV jobs; use -1 for all CPUs")
    parser.add_argument(
        "--no-undersampling",
        dest="undersample",
        action="store_false",
        help="Keep the original training distribution instead of undersampling",
    )
    parser.set_defaults(undersample=True)
    args = parser.parse_args()
    if not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1")
    if args.cv < 2:
        parser.error("--cv must be at least 2")
    return args


if __name__ == "__main__":
    run(parse_args())
