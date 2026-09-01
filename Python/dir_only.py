#!/usr/bin/env python

"""
This script applies the Disparate Impact Remover (DIR) pre-processing
technique to the COMPAS dataset to mitigate bias before training a
logistic regression classifier. The performance and fairness metrics
are then evaluated on the test set.
"""
from sklearn.linear_model import LogisticRegression
from aif360.algorithms.preprocessing import DisparateImpactRemover
from common.data_utils import load_raw_compas, to_binary_label_dataset, train_val_test_split
from common.metrics_utils import print_metrics
from config.fairness_config import (
    RANDOM_STATE
)

def main():
    
    # Load and prepare the COMPAS dataset.
    df = load_raw_compas()
    bld = to_binary_label_dataset(df)
    train, _, test = train_val_test_split(bld)

    # Apply Disparate Impact Remover (DIR) pre-processing
    # repair_level=1.0 applies full fairness repair (strongest debiasing)
    dir_ = DisparateImpactRemover(repair_level=1.0)
    train_repaired = dir_.fit_transform(train)

    # Train on dir(repaired) train set, test set stays the same.
    # train.labels (from AIF360) is typically stored as a 2D column array.
    # scikit-learn expects labels to be a flat 1D array.
    # .ravel() is used flatten the array.
    X_train, y_train = train_repaired.features, train_repaired.labels.ravel()
    X_test, _ = test.features, test.labels.ravel()

    # Train Logistic Regression model on repaired data.
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)

    # Format predictions for fairness evaluation.
    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)[:, 1]
    dir_pred = test.copy()
    dir_pred.labels = y_pred.reshape(-1, 1)
    dir_pred.scores = y_score.reshape(-1, 1)

    # Evaluate fairness metrics for this model.
    print_metrics(
        dataset_true=test,
        dataset_pred=dir_pred
    )

if __name__ == "__main__":
    main()
