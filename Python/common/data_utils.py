#!/usr/bin/env python

"""
Data utilities for COMPAS fairness experiments.

Includes functions to 
    - load raw data
    - convert to AIF360 BinaryLabelDataset,
    - split into train/val/test sets.
"""

import pandas as pd
from aif360.datasets import BinaryLabelDataset
from config.fairness_config import (
    PRED_COL, PROT_COL, PRIV_COL, LABEL_COL,
    RANDOM_STATE
)

def load_raw_compas(path="../Dataset/propublica_data_for_fairml.csv", drop_score=True):
    """
    Load and preprocess the COMPAS dataset.

    Parameters:
        path : str
            Path to the COMPAS CSV file.
        drop_score : bool
            If True, removes the COMPAS model prediction (`score_factor`)
            to prevent data leakage when training ML models.
            If False, keeps the score for fairness auditing of COMPAS itself.

    Returns:
        df : pandas.DataFrame
            Cleaned and formatted COMPAS dataset.
    """
    df = pd.read_csv(path)

    # Race columns in the dataset.
    race_cols = ["African_American", "Asian", "Hispanic", "Native_American", "Other"]

    # Create White column (1 = White, 0 = non-White)
    df["White"] = (df[race_cols].sum(axis=1) == 0).astype(int)
    
    # Ensure relevant columns are integers
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    df[PRED_COL] = df[PRED_COL].astype(int)
    df[PROT_COL] = df[PROT_COL].astype(int)
    df[PRIV_COL] = df[PRIV_COL].astype(int)

    if drop_score:
        # Drop existing score_factor if present to avoid leakage
        df = df.drop(columns=[PRED_COL], errors="ignore")  

    return df


def to_binary_label_dataset(df):
    """
    Convert a pandas DataFrame into an AIF360 BinaryLabelDataset.
    This enables fairness-aware metrics and mitigation algorithms.

    Parameters:
        df : pandas.DataFrame
            Preprocessed COMPAS dataset.
    Returns:
        BinaryLabelDataset object
    """
    return BinaryLabelDataset(
        favorable_label=0, # Did NOT reoffend within 2 years.
        unfavorable_label=1, # Reoffended within 2 years.
        df=df,
        label_names=[LABEL_COL],
        protected_attribute_names=[PROT_COL, PRIV_COL]
    )

def train_val_test_split(bld, test_size=0.3, val_size=0.2):
    """
    Split BinaryLabelDataset into train/val/test using index splits.
    val_size is a fraction of the *train* portion.


    Parameters:
        bld : BinaryLabelDataset
            Full dataset to split.
        test_size : float
            Fraction of dataset reserved for testing.
        val_size : float
            Fraction of the remaining train+val split used for validation.

    Returns:
        train, val, test : BinaryLabelDataset
            Three reproducible dataset splits.
    
    Train: 56%
    Val: 14%
    Test: 30%
    """
    # First split train+val vs test
    train_val, test = bld.split([1 - test_size], shuffle=True, seed=RANDOM_STATE)

    # Now split train vs val inside train_val 
    n_train_val = train_val.features.shape[0]
    n_train = int((1 - val_size) * n_train_val)
    idx = list(range(n_train_val))
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    train = train_val.subset(train_idx)
    val = train_val.subset(val_idx)
    return train, val, test
