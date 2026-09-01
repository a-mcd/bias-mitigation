#!/usr/bin/env python

"""
This script trains a simple logistic regression model on the cleaned COMPAS dataset.
The purpose of this baseline model is to provide a comparison point for fairness
analysis.
"""

from sklearn.linear_model import LogisticRegression
from common.data_utils import load_raw_compas, to_binary_label_dataset, train_val_test_split
from common.metrics_utils import print_metrics
from config.fairness_config import (
    RANDOM_STATE
)

def main():

    # Load and prepare the COMPAS dataset.
    df = load_raw_compas()
    bld = to_binary_label_dataset(df)

    # Split into train, validation, and test sets (validation is unused here).
    train, _, test = train_val_test_split(bld)

    # Prepare feature and label arrays.
    # train.labels (from AIF360) is typically stored as a 2D column array.
    # scikit-learn expects labels to be a flat 1D array.
    # .ravel() is used flatten the array.
    X_train, y_train = train.features, train.labels.ravel()
    X_test, _   = test.features, test.labels.ravel()

    # Train a baseline Logistic Regression model.
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)

    # Format predictions and scores into a BinaryLabelDataset.
    baseline_pred = test.copy()
    baseline_pred.labels = clf.predict(X_test).reshape(-1, 1)
    baseline_pred.scores = clf.predict_proba(X_test)[:, 1].reshape(-1, 1)

    # Evaluate fairness metrics for this model.
    print_metrics(
        dataset_true=test,
        dataset_pred=baseline_pred
    )
    

if __name__ == "__main__":
    main()
