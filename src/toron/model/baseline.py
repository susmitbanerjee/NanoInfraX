from pathlib import Path
import joblib
import polars as pl

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def train():
    train_df = pl.read_parquet("../datasets/processed/train.parquet")
    val_df = pl.read_parquet("../datasets/processed/validation.parquet")

    model = Pipeline([
        ("vectorizer", TfidfVectorizer( analyzer="char", ngram_range=(2, 5), sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    ])

    model.fit(train_df["text"].to_list(), train_df["cause"].to_list())

    predictions = model.predict(val_df["text"].to_list())

    print(classification_report(val_df["cause"].to_list(), predictions, zero_division=0))

    path = Path("../models/baseline")
    path.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, "../models/model.joblib")

if __name__ == "__main__":
    train()