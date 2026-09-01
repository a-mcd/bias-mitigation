#!/usr/bin/env python

"""
Fairness configuration used across the project.
This file defines a single source of truth for protected and privileged attributes.
"""

PROT_COL = "African_American"   # 1 = African-American (unprivileged group)
PRIV_COL = "White"              # 1 = White (privileged group)

# Ground-truth label column
# 0 = Did NOT reoffend within 2 years (favourable outcome)
# 1 = Reoffended within 2 years (unfavourable outcome)
LABEL_COL = "Two_yr_Recidivism"

# Column containing predictions when evaluating COMPAS fairness
# Represents the assigned COMPAS binary risk score (0 = low, 1 = medium/high)
PRED_COL = "score_factor"

# AIF360 format: grouping structures indicating which samples belong
# to privileged vs unprivileged groups for fairness metric calculation.
PRIVILEGED_GROUPS = [{PRIV_COL: 1}]
UNPRIVILEGED_GROUPS = [{PROT_COL: 1}]

# For reproducibility
RANDOM_STATE = 42 
