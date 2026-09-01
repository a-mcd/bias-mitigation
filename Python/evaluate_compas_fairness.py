#!/usr/bin/env python

"""
Purpose:
--------
This script evaluates the fairness of the original COMPAS risk score_factor
using the AIF360 fairness metrics framework.

It loads the raw COMPAS dataset, converts it into a BinaryLabelDataset format,
assigns the COMPAS risk score as the prediction source, and computes key fairness
metrics (including disparate impact, demographic parity, equalized odds, and
predictive parity) to assess whether the COMPAS system treats protected and
privileged groups differently.

This file does not train a model, it performs a fairness audit of the
existing COMPAS scoring system.
"""


from common.data_utils import load_raw_compas, to_binary_label_dataset
from common.metrics_utils import print_metrics
from config.fairness_config import (PRED_COL)

# Load dataset and create/format necessary columns
df = load_raw_compas(drop_score=False)

dataset_true = to_binary_label_dataset(df)

# AIF360 requires two datasets when computing fairness metrics:
# 1) dataset_true: contains true labels, the actual outcomes
# 2) dataset_pred: contains predicted labels from the model
# Gets the column containing the predictions and reshapes it into a 2D column vector required for AIF360
dataset_pred = dataset_true.copy()
dataset_pred.labels = df[PRED_COL].values.reshape(-1, 1)


# Print fairness metrics
print_metrics(dataset_true, dataset_pred)