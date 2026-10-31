# AI Sentiment Classifier

A small, reproducible machine-learning project that classifies short text as **positive** or **negative**.

The project uses a scikit-learn `Pipeline` with:

1. `TfidfVectorizer` for converting text into numerical features.
2. `LogisticRegression` for binary classification.

The training dataset is intentionally tiny and embedded in the source so the example runs offline without downloading private or third-party data. It is designed as a learning and connector-test project, not as a production-grade sentiment model.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python -m src.sentiment_classifier "The support was fast and helpful"
```

Example output:

```text
positive (confidence=0.73)
```

Run the smoke tests:

```bash
python -m unittest discover -s tests -v
```

## Project layout

- `src/sentiment_classifier.py` — training data, model pipeline, and CLI.
- `tests/test_sentiment_classifier.py` — basic prediction tests.
- `requirements.txt` — pinned runtime dependency.

## Limitations

The included dataset has only a few handcrafted examples. Real applications should use a larger, representative, reviewed dataset and evaluate accuracy, calibration, bias, and drift before deployment.

## License and attribution

The original project code is released under the MIT License. It uses [scikit-learn](https://scikit-learn.org/), which is distributed under the BSD 3-Clause License. See `LICENSE` for this project's license and the official [scikit-learn license](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) for dependency terms.
