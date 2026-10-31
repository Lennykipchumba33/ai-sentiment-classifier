"""A tiny offline sentiment classifier built with scikit-learn."""

from __future__ import annotations

import argparse
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


TRAINING_TEXTS = [
    "I love this product",
    "The experience was excellent",
    "This is wonderful and easy to use",
    "Fantastic support and fast delivery",
    "I am very happy with the result",
    "The app is reliable and helpful",
    "I hate this product",
    "The experience was terrible",
    "This is disappointing and difficult to use",
    "Awful support and very late delivery",
    "I am unhappy with the result",
    "The app is buggy and frustrating",
]

TRAINING_LABELS = [
    "positive",
    "positive",
    "positive",
    "positive",
    "positive",
    "positive",
    "negative",
    "negative",
    "negative",
    "negative",
    "negative",
    "negative",
]


def build_model() -> Pipeline:
    """Create an unfitted text-classification pipeline."""
    return Pipeline(
        [
            ("features", TfidfVectorizer(lowercase=True, ngram_range=(1, 2))),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def train_model() -> Pipeline:
    """Train and return the demo model."""
    model = build_model()
    model.fit(TRAINING_TEXTS, TRAINING_LABELS)
    return model


def predict(texts: Iterable[str]) -> list[tuple[str, float]]:
    """Return (label, confidence) pairs for the supplied texts."""
    model = train_model()
    inputs = list(texts)
    probabilities = model.predict_proba(inputs)
    labels = model.predict(inputs)
    return [
        (label, float(probabilities[index].max()))
        for index, label in enumerate(labels)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="+", help="Text to classify")
    args = parser.parse_args()

    for text, (label, confidence) in zip(args.text, predict(args.text)):
        print(f"{label} (confidence={confidence:.2f}) :: {text}")


if __name__ == "__main__":
    main()
