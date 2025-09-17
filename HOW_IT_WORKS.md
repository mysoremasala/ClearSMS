# How This Project Works

This document explains the end‑to‑end flow of the SMS Spam Detector, including data handling, preprocessing, model training, inference, and the web API/UI behavior.

## Overview
- The app is a Flask server (`app.py`) exposing:
  - `/` to serve the HTML frontend (`templates/index.html` + `static/`)
  - `/predict` to classify a provided text
  - `/model-status` to expose training/artifact status
- On startup, the server tries to load a previously trained model and vectorizer from `model/`.
- If artifacts are missing, it loads the dataset (if found) or uses built‑in sample data and trains a new model, then saves the artifacts.

## Data Flow
1. On boot, `load_and_train_model()` runs:
   - Searches for dataset at `model/SMSSpamCollection` (TSV with columns: `label`, `text`).
   - If present, loads it with `pandas.read_csv(..., sep='\t', header=None, names=['label','text'])`.
   - Else, creates a repeated small sample dataset for demonstration.

2. Preprocessing via `preprocess_text(text)`:
   - Lowercases the text
   - Removes punctuation and numbers using a regex (`[^a-zA-Z\s]`)
   - Collapses multiple spaces and trims

3. Label encoding:
   - Maps `ham -> 0`, `spam -> 1` into a new column `label_binary`.

4. Train/test split:
   - `train_test_split(..., test_size=0.2, stratify=y, random_state=42)` to keep class balance.

5. Vectorization using TF‑IDF (`sklearn.feature_extraction.text.TfidfVectorizer`):
   - Key params: `max_features=5000`, `stop_words='english'`, `ngram_range=(1,2)`, `min_df=2`, `max_df=0.95`, `sublinear_tf=True`.
   - Fit on train set, transform both train and test sets.

6. Model training using Logistic Regression (`sklearn.linear_model.LogisticRegression`):
   - Params: `class_weight='balanced'`, `max_iter=1000`, `random_state=42`.
   - Trains on vectorized train data.

7. Evaluation:
   - Computes `accuracy_score` and `f1_score` on the test set for sanity check (printed to console).

8. Persistence (artifacts):
   - Saves the trained model as `model/spam_model.joblib` and vectorizer as `model/vectorizer.joblib` using `joblib.dump`.

## Inference Path
1. Client enters/pastes text in the UI.
2. Browser JS sends `POST /predict` with JSON `{ message: "..." }`.
3. Server handler:
   - Validates input (non‑empty, length ≤ 1000).
   - Calls `predict_spam(text)`:
     - Applies `preprocess_text`.
     - Transforms via loaded `vectorizer`.
     - Gets probabilities and predicted class from `model`.
   - Returns JSON: `{ prediction: 'Spam'|'Ham', confidence: number, message: truncated }`.
4. UI updates the result card, confidence bar, and explanation text.

## Frontend Details (`templates/index.html` + `static/style.css`)
- Uses a clean card layout with Font Awesome icons.
- JavaScript handles:
  - Character counting and color ramp
  - Sample message buttons
  - Calling `/predict` and rendering results/errors
  - Checking `/model-status` on load to show Ready/Training/Error

## Alternative: Standalone Training Script (`model/spam_classifier.py`)
- Running this script directly will:
  - Load dataset from current working directory (`SMSSpamCollection` expected next to the script when invoked directly), or build a small sample
  - Train the same TF‑IDF + Logistic Regression pipeline
  - Save `spam_model.joblib` and `vectorizer.joblib` in the current working directory
- Note: The Flask app expects artifacts under `model/`. For consistency, prefer starting the Flask app so artifacts are stored in `model/`.

## Operational Notes
- Determinism: `random_state=42` for reproducibility in splitting/training.
- Class imbalance: handled via `class_weight='balanced'`.
- Resource usage: TF‑IDF with up to 5000 features; training happens once and artifacts are cached.
- Error handling: basic try/except with JSON error responses for `/predict`.

## Extending the Project
- Swap classifier (e.g., Linear SVM, Naive Bayes) and compare metrics.
- Add model versioning and a background training job.
- Add more robust text normalization (URLs, emojis), lemmatization.
- Deploy behind a production server (e.g., gunicorn + reverse proxy) and containerize.


