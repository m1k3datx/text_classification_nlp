import pandas as pd
from sklearn.model_selection import train_test_split

from text_classification import load_dataset


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
    train, holdout = train_test_split(
        data, test_size=0.5, random_state=123, stratify=data["relevance"]
    )
    assert set(train["text"]).isdisjoint(set(holdout["text"]))


def test_conflicting_duplicate_labels_are_excluded(tmp_path):
    path = tmp_path / "articles.csv"
    pd.DataFrame(
        {
            "text": ["Same article", "same <br/> article", "Unique yes", "Unique no"],
            "relevance": ["yes", "no", "yes", "no"],
        }
    ).to_csv(path, index=False)

    data = load_dataset(path)

    assert set(data["text"]) == {"unique yes", "unique no"}


def test_normalization_removes_documented_stop_words(tmp_path):
    path = tmp_path / "articles.csv"
    pd.DataFrame(
        {
            "text": ["The market and the economy"],
            "relevance": ["yes"],
        }
    ).to_csv(path, index=False)

    data = load_dataset(path)

    assert data.iloc[0]["text"] == "market economy"
