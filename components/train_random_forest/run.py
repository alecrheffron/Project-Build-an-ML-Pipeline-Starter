import argparse
import json
import joblib
import numpy as np
import pandas as pd
import wandb
import mlflow.sklearn
import os
import shutil

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer

TARGET = "price"

# Common NYC Airbnb columns used in this project
TEXT_COL = "name"
CAT_COLS = ["neighbourhood_group", "neighbourhood", "room_type"]
NUM_COLS = [
    "latitude",
    "longitude",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
]


def _existing(cols, df):
    return [c for c in cols if c in df.columns]

def squeeze_1d(x):
    """Convert (n_samples, 1) array to (n_samples,) for TfidfVectorizer."""
    return x.ravel()

def go(args):
    run = wandb.init(job_type="train_random_forest")

    # Load datasets from W&B artifacts
    train_art = run.use_artifact(args.train_artifact)
    test_art = run.use_artifact(args.test_artifact)
    train_path = train_art.file()
    test_path = test_art.file()

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Defensive: ensure target exists
    if TARGET not in train_df.columns or TARGET not in test_df.columns:
        raise ValueError(f"Expected target column '{TARGET}' in both train and test data")

    # Split features/target
    y_train = train_df[TARGET].astype(float)
    y_test = test_df[TARGET].astype(float)

    X_train = train_df.drop(columns=[TARGET]).copy()
    X_test = test_df.drop(columns=[TARGET]).copy()

    # Column lists that actually exist
    cat_cols = _existing(CAT_COLS, X_train)
    num_cols = _existing(NUM_COLS, X_train)
    has_text = TEXT_COL in X_train.columns

    # Preprocess pipelines with imputers (RF cannot handle NaNs)
    text_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="")),
        ("squeeze", FunctionTransformer(squeeze_1d, validate=False)),
        ("tfidf", TfidfVectorizer(max_features=args.max_tfidf_features)),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    transformers = []

    if has_text:
        transformers.append(("text", text_pipe, [TEXT_COL]))

    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))

    if num_cols:
        transformers.append(("num", num_pipe, num_cols))

    if not transformers:
        raise ValueError("No usable feature columns found. Check your cleaned/split dataset columns.")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    # Load RF config JSON
    with open(args.rf_config, "r") as f:
        rf_params = json.load(f)

    # Log RF hyperparams to W&B so they show up as table columns
    run.config.update({k: v for k, v in rf_params.items()}, allow_val_change=True)

    # Optional: make sure these are definitely present as top-level keys
    run.config.update(
        {
            "max_depth": rf_params.get("max_depth", None),
            "n_estimators": rf_params.get("n_estimators", None),
            "max_features": rf_params.get("max_features", None),
            "min_samples_split": rf_params.get("min_samples_split", None),
            "min_samples_leaf": rf_params.get("min_samples_leaf", None),
        },
        allow_val_change=True,
    )
    model = RandomForestRegressor(**rf_params)

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])

    # Train
    pipe.fit(X_train, y_train)

    # Evaluate
    preds = pipe.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    run.log({"rmse": rmse, "mae": mae, "r2": r2})

    # Save model locally
    model_dir = "random_forest_model"

    # clean up if rerunning locally
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)

    # This creates MLmodel + conda.yaml/requirements + model.pkl etc.
    mlflow.sklearn.save_model(pipe, model_dir)

    out_art = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    model_dir = "random_forest_model"

    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)

    mlflow.sklearn.save_model(pipe, model_dir)

    out_art = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )

    # Add files so MLmodel ends up at the artifact ROOT
    for root, _, files in os.walk(model_dir):
        for fname in files:
            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, model_dir)
            out_art.add_file(full_path, name=rel_path)

    run.log_artifact(out_art)

    run.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_artifact", required=True)
    p.add_argument("--test_artifact", required=True)
    p.add_argument("--rf_config", required=True)
    p.add_argument("--max_tfidf_features", type=int, required=True)
    p.add_argument("--output_artifact", required=True)
    p.add_argument("--output_type", required=True)
    p.add_argument("--output_description", required=True)
    go(p.parse_args())