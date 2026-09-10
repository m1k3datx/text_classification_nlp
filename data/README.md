
# Dataset notes

`US-Economic-News.csv` is a historical sample of news articles with
manually assigned `relevance` labels (`yes`, `no`, or `not sure`). The workflow
excludes `not sure` rows because the task is binary classification. The file
is third-party content, not original project code.

The repository does not assert ownership of the article text, guarantee that
the source publisher permits redistribution, or provide a license for the
dataset. Before redistributing, publishing, or using the data commercially,
identify the original source and verify applicable copyright, database,
terms-of-use, and privacy obligations. The MIT license in the repository root
covers project code only.

Labels are historical annotations and may contain ambiguity or noise. They
should not be treated as authoritative economic classification. The workflow
normalizes text and removes duplicate normalized articles (and conflicting
duplicate labels) before splitting so the same article cannot appear in
training and evaluation partitions.
