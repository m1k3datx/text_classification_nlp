import pandas as pd
from sklearn.model_selection import train_test_split

import pytest

from text_classification import clean_text, load_dataset


def test_duplicate_normalized_texts_are_removed_before_splitting(tmp_path):
    path = tmp_path / "articles.csv"
    pd.DataFrame(
        {
            "text": [
                "Same <br/> article!",
                "same article",
                "Unique relevant article",
                "Another relevant story",
                "Unique irrelevant article",
                "Another irrelevant story",
            ],
            "relevance": ["yes", "yes", "yes", "yes", "no", "no"],
        }
    ).to_csv(path, index=False)

    data = load_dataset(path)

    assert len(data) == 5
    assert data["text"].is_unique
    train, holdout = train_test_split(
        data, test_size=0.5, random_state=123, stratify=data["relevance"]
    )
    assert set(train["text"]).isdisjoint(set(holdout["text"]))


def test_conflicting_duplicate_labels_are_excluded(tmp_path):
    path = tmp_path / "articles.csv"
    pd.DataFrame(
        {
            "text": [
                "Same article",
                "same <br/> article",
                "Unique positive",
                "Unique negative",
            ],
            "relevance": ["yes", "no", "yes", "no"],
        }
    ).to_csv(path, index=False)

    data = load_dataset(path)

    assert set(data["text"]) == {"unique positive", "unique negative"}
    assert "same article" not in set(data["text"])


def test_clean_text_removes_documented_stop_words():
    assert clean_text("The market and the economy") == "market economy"


def test_loader_accepts_both_binary_labels(tmp_path):
    path = tmp_path / "articles.csv"
    pd.DataFrame(
        {
            "text": ["Positive article", "Negative article"],
            "relevance": ["yes", "no"],
        }
    ).to_csv(path, index=False)

    data = load_dataset(path)

    assert set(data["relevance"]) == {0, 1}


def test_loader_rejects_single_class_dataset(tmp_path):
    path = tmp_path / "articles.csv"
    pd.DataFrame(
        {
            "text": ["Only positive article"],
            "relevance": ["yes"],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="both 'yes' and 'no'"):
        load_dataset(path)
