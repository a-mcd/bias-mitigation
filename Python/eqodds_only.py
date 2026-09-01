#!/usr/bin/env python

"""
This script trains a Logistic Regression classifier on the COMPAS dataset,
then applies Equalized Odds post-processing from AIF360 to correct disparities
across protected groups.
"""

from sklearn.linear_model import LogisticRegression
from aif360.algorithms.postprocessing import EqOddsPostprocessing
from common.metrics_utils import print_metrics
from common.data_utils import (
    load_raw_compas, to_binary_label_dataset, train_val_test_split
)
from config.fairness_config import (
    PRIVILEGED_GROUPS, UNPRIVILEGED_GROUPS,
    RANDOM_STATE
)

def main():
    
    # Load and prepare the COMPAS dataset.
    df = load_raw_compas()
    bld = to_binary_label_dataset(df)
    train, val, test = train_val_test_split(bld)

    # train.labels (from AIF360) is typically stored as a 2D column array.
    # scikit-learn expects labels to be a flat 1D array.
    # .ravel() is used flatten the array.
    X_train, y_train = train.features, train.labels.ravel()
    X_val, _ = val.features, val.labels.ravel()
    X_test, _ = test.features, test.labels.ravel()

    # Train Logistic Regression model.
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)

    # Equalized Odds requires 
    #  - the original dataset structure (features, protected attributes, labels)
    #  - a modified copy containing predicted labels
    
    # val_pred will hold the Logistic Regression’s predictions on the validation set.
    # Used by Equalized Odds to learn the adjustments.
    val_pred = val.copy()
    val_pred.labels = clf.predict(X_val).reshape(-1, 1)
    val_pred.scores = clf.predict_proba(X_val)[:, 1].reshape(-1, 1)

    # test_pred is the Logistic Regression’s predictions on the test set.
    # After Equalized Odds is fitted on the validation set, it will adjust these predictions
    test_pred = test.copy()
    test_pred.labels = clf.predict(X_test).reshape(-1, 1)
    test_pred.scores = clf.predict_proba(X_test)[:, 1].reshape(-1, 1)

    # Fit Equalized Odds on val set
    #  learns adjustments to reduce the disparity
    eq = EqOddsPostprocessing(
        privileged_groups=PRIVILEGED_GROUPS,
        unprivileged_groups=UNPRIVILEGED_GROUPS,
        seed=RANDOM_STATE
    )
    eq = eq.fit(val, val_pred)

    # Apply the EO adjustments to test predictions
    eq_test_pred = eq.predict(test_pred)

    # Evaluate fairness metrics for this model.
    print_metrics(
        dataset_true=test,
        dataset_pred=eq_test_pred
    )

if __name__ == "__main__":
    main()
