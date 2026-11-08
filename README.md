# AI Sentiment Classifier

[![CI](https://github.com/Lennykipchumba33/ai-sentiment-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/Lennykipchumba33/ai-sentiment-classifier/actions/workflows/ci.yml)

A small, reproducible NLP project that classifies short text as **positive** or **negative**. It demonstrates a complete offline scikit-learn pipeline: TF-IDF features, logistic regression, a command-line interface, tests, and explicit limitations.

## What this demonstrates

- Text preprocessing with `TfidfVectorizer`, including unigram and bigram features.
- A deterministic `LogisticRegression` classifier wrapped in a scikit-learn `Pipeline`.
- Confidence scores from `predict_proba`.
- A simple CLI and unit tests that can run without downloading data.
- Honest model documentation: the embedded dataset is intentionally tiny and is not suitable for production decisions.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python -m src.sentiment_classifier \
  "The support was fast and helpful" \
  "The replacement process was frustrating"
python -m unittest discover -s tests -v
```

Example output:

```text
positive (confidence=0.73) :: The support was fast and helpful
negative (confidence=0.67) :: The replacement process was frustrating
```

## Architecture

```text
text input -> TF-IDF vectorizer -> logistic regression -> label + confidence
```

The model is trained in memory from a small, transparent set of labeled examples. There are no hidden downloads, API keys, or external services.

## Project layout

- `src/sentiment_classifier.py` — training data, model pipeline, and CLI.
- `tests/test_sentiment_classifier.py` — unit tests and realistic examples.
- `.github/workflows/ci.yml` — automated test and compile checks.
- `requirements.txt` — runtime dependency.

## Limitations and next steps

The dataset contains only a few handcrafted examples. A production system would require a larger representative dataset, a held-out evaluation set, calibration checks, bias analysis, drift monitoring, and a documented review process. Useful next steps include adding a dataset adapter, cross-validation, model serialization, and an inference API.

## License and attribution

The project is released under the MIT License. See `LICENSE`. It uses [scikit-learn](https://scikit-learn.org/), distributed under the BSD 3-Clause License.
