#!/usr/bin/env python

"""
Metric utilities for AIF360.
"""
import numpy as np
from aif360.metrics import ClassificationMetric
from config.fairness_config import (
    PROT_COL, PRIV_COL,
    PRIVILEGED_GROUPS, UNPRIVILEGED_GROUPS
)

def print_metrics(dataset_true, dataset_pred):
    """
    Compute and print a standard set of accuracy and fairness metrics
    for a given model prediction vs. actual truth.

    Parameters:
        dataset_true : BinaryLabelDataset
            Contains the true outcomes.
        
        dataset_pred : BinaryLabelDataset
            Contains the predicted outcomes from a model or COMPAS score.

    """

    # Create metric object which internally compares predictions vs actual truth.
    metric = ClassificationMetric(
        dataset_true,
        dataset_pred,
        privileged_groups=PRIVILEGED_GROUPS,     # Non protected attribute
        unprivileged_groups=UNPRIVILEGED_GROUPS    # Protected attribute
    )

    accuracy = metric.accuracy()
    print("\nModel Accuracy:", round(accuracy, 3))

    print("\nFairness Metrics\n")
    print(f"Privileged Group: {PRIV_COL}")
    print(f"Unprivileged Group: {PROT_COL}")

    # -----------------------------
    # Demographic Parity & DI
    # -----------------------------
    dpd = metric.statistical_parity_difference()
    di  = metric.disparate_impact()

    print("\nDemographic Parity Difference:", round(dpd, 3))
    print("Disparate Impact (Ratio):", round(di, 3))

    # -----------------------------
    # Equalized Odds: TPR/FPR diffs
    # -----------------------------
    tpr_diff = metric.true_positive_rate_difference()
    fpr_diff = metric.false_positive_rate_difference()

    print("\nEqualized Odds (should ideally ≈ 0):")
    print(" - True Positive Rate difference:", round(tpr_diff, 3))
    print(" - False Positive Rate difference:", round(fpr_diff, 3))

    # -----------------------------
    # Equality of Opportunity
    # -----------------------------
    eopp_diff = metric.equal_opportunity_difference()
    print("\nEquality of Opportunity (TPR diff):", round(eopp_diff, 3))

    # -----------------------------
    # Overall Predictive Parity
    # -----------------------------
    print("\nOverall Predictive Parity:")

    # PPV comparison
    ppv_priv   = metric.positive_predictive_value(privileged=True)
    ppv_unpriv = metric.positive_predictive_value(privileged=False)
    ppv_diff = ppv_unpriv - ppv_priv
    print("  - PPV (Privileged):", round(ppv_priv, 3))
    print("  - PPV (Unprivileged):", round(ppv_unpriv, 3))
    print("  - PPV Diff:", round(ppv_diff, 3))

    # NPV comparison
    npv_priv   = metric.negative_predictive_value(privileged=True)
    npv_unpriv = metric.negative_predictive_value(privileged=False)
    npv_diff = npv_unpriv - npv_priv
    print("  - NPV (Privileged):", round(npv_priv, 3))
    print("  - NPV (Unprivileged):", round(npv_unpriv, 3))
    print("  - NPV Diff:", round(npv_diff, 3))


    # Check if scores are available for calibration metrics
    scores = getattr(dataset_pred, "scores", None)
    if scores is None:
        print("\n[Calibration & balance metrics skipped: 'dataset_pred.scores' is not set]")
        return

    scores_arr = np.asarray(scores).ravel()
    unique_vals = np.unique(scores_arr)

    # If scores are only 0/1 (hard labels), not probabilities, skip.
    if unique_vals.size <= 2 and set(unique_vals).issubset({0.0, 1.0}):
        print("\n[Calibration & balance metrics skipped: 'dataset_pred.scores' appears to be hard labels (0/1), not probabilities]")
        return
    
    # ----------------------------------------------------
    # Calibration & Balance metrics (score-based)
    # ----------------------------------------------------
    
    # Flatten arrays
    y_true = dataset_true.labels.ravel().astype(int)
    y_score = scores.ravel().astype(float)

    # Protected attributes (order: [PROT_COL, PRIV_COL])
    prot_attr = dataset_true.protected_attributes
    #print(dataset_true.protected_attribute_names)
    prot_flag = prot_attr[:, 0].astype(int)  # e.g. African_American
    priv_flag = prot_attr[:, 1].astype(int)  # e.g. White

    # Keep only resutls which belong to privileged or unprivileged groups
    valid_mask = (prot_flag == 1) | (priv_flag == 1)
    y_true_g = y_true[valid_mask]
    y_score_g = y_score[valid_mask]
    prot_flag_g = prot_flag[valid_mask]
    priv_flag_g = priv_flag[valid_mask]

    # Create boolean array marking who belongs to unpriv and priv_mask groups.
    unpriv_mask = prot_flag_g == 1
    priv_mask = priv_flag_g == 1

    # ----------------------------------------------------
    # Balance metrics
    # ----------------------------------------------------

    # Positive class: y_true == 1 (here: reoffended within 2 years)
    print("\nBalance Metrics:")

    # Positive class: y_true == 1 (here: reoffended within 2 years)
    pos_mask = y_true_g == 1
    if np.any(pos_mask & unpriv_mask) and np.any(pos_mask & priv_mask):
        # Compute average scores for positive class in each group.
        avg_score_pos_unpriv = y_score_g[pos_mask & unpriv_mask].mean()
        avg_score_pos_priv = y_score_g[pos_mask & priv_mask].mean()
        # Compute the difference.
        diff_pos = avg_score_pos_unpriv - avg_score_pos_priv

        print("  - Balance for the Positive Class:")
        print(f"      * Avg score (Unprivileged, {PROT_COL}):", round(avg_score_pos_unpriv, 4))
        print(f"      * Avg score (Privileged,  {PRIV_COL}):", round(avg_score_pos_priv, 4))
        print("      * Difference (Unpriv - Priv):", round(diff_pos, 4))
    else:
        print("  - [Balance for the Positive Class skipped: insufficient positive samples in one group]")

    
    # Negative class: y_true == 0 (here: did NOT reoffend)
    neg_mask = y_true_g == 0
    if np.any(neg_mask & unpriv_mask) and np.any(neg_mask & priv_mask):
        # Compute average scores for negative class in each group.
        avg_score_neg_unpriv = y_score_g[neg_mask & unpriv_mask].mean()
        avg_score_neg_priv = y_score_g[neg_mask & priv_mask].mean()
        # Compute the difference.
        diff_neg = avg_score_neg_unpriv - avg_score_neg_priv

        print("  - Balance for the Negative Class:")
        print(f"      * Avg score (Unprivileged, {PROT_COL}):", round(avg_score_neg_unpriv, 4))
        print(f"      * Avg score (Privileged,   {PRIV_COL}):", round(avg_score_neg_priv, 4))
        print("      * Difference (Unpriv - Priv):", round(diff_neg, 4))
    else:
        print("  - [Balance for the Negative Class skipped: insufficient negative samples in one group]")


    # ----------------------------------------------------
    # Calibration metrics
    # ----------------------------------------------------
    # Inspired by reference 20
    print(f"\nCalibration bins (Unprivileged group – {PROT_COL}):")
    bins_unpriv = _calibration_bins(
        y_true_g[unpriv_mask],
        y_score_g[unpriv_mask],
    )
    for b in bins_unpriv:
        print(f"  Bin [{b['lower']:.1f}, {b['upper']:.1f}]: "
              f"count={b['count']}, "
              f"avg_score={b['avg_score']:.3f}, "
              f"likelihood_pos={b['likelihood_pos']:.3f}")
    
    print(f"\nCalibration bins (Privileged group – {PRIV_COL}):")
    bins_priv = _calibration_bins(
        y_true_g[priv_mask],
        y_score_g[priv_mask],
    )
    for b in bins_priv:
        print(f"  Bin [{b['lower']:.1f}, {b['upper']:.1f}]: "
              f"count={b['count']}, "
              f"avg_score={b['avg_score']:.3f}, "
              f"likelihood_pos={b['likelihood_pos']:.3f}")
    
    _compare_group_calibration(bins_unpriv, bins_priv, PROT_COL, PRIV_COL)




def _calibration_bins(y_true, y_score, n_bins=10):
    """
    Parameters:
        y_true : actual labels (0 or 1)
        y_score: predicted probabilities for the positive class (Two_yr_Recidivism = 1)
        n_bins : number of bins
    
    Returns:
        bins_stats : list of dict
            A list of length `n_bins` where each element contains summary statistics
            for one bin. Each dictionary has the following keys:

            - 'lower': float  
                Lower edge of the bin.
            - 'upper': float  
                Upper edge of the bin.
            - 'count': int  
                Number of samples falling into the bin.
            - 'avg_score': float  
                Mean predicted probability for samples in the bin. Represents the model's average confidence.
            - 'likelihood_pos': float  
                Fraction of samples in the bin that belong to the positive class i.e., true rate of Two_yr_Recidivism = 1.

            These outputs allow comparison between model confidence and actual outcomes for calibration analysis.
    """

    # Convert inputs into NumPy arrays.
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    # Create bins and assign each score to a bin.
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_score, bin_edges, right=True)

    bins_stats = []
    for i in range(1, n_bins + 1):

        in_bin = (bin_indices == i)
        count = in_bin.sum()

        if count > 0:
            # Compute average predicted score in this bin
            avg_score = y_score[in_bin].mean()
            # Compute empirical likelihood of belonging to the positive class
            likelihood_pos = y_true[in_bin].mean()
        else:
            avg_score = np.nan
            likelihood_pos = np.nan


        # Store results
        bins_stats.append({
            'lower': bin_edges[i - 1],
            'upper': bin_edges[i],
            'count': in_bin.sum(),
            # Average predicted score in this bin.
            # What the model thinks is the risk level for people in this bin.
            'avg_score': avg_score, 
            # Probability of belonging to the positive class - recidivism within 2 years
            'likelihood_pos': likelihood_pos,
        })

    return bins_stats


def _compare_group_calibration(bins_unpriv, bins_priv, unpriv_name, priv_name):
    """
    Compare likelihood of positive class per bin between two groups.

    Assumes both bins_* were built using the same bin_edges and n_bins,
    so entries at the same index refer to the same [lower, upper] bin.

    Parameters:
        bins_unpriv : list of dict
            Calibration bin statistics for the unprivileged group.
            Each dict must contain:
                - 'lower': float, lower edge of the bin
                - 'upper': float, upper edge of the bin
                - 'count': int, number of samples in the bin
                - 'avg_score': float, mean predicted probability
                - 'likelihood_pos': float, empirical probability of positive class
        bins_priv : list of dict
            Calibration bin statistics for the privileged group, with the same
            structure and bin edges as `bins_unpriv`.
        unpriv_name : str
            Name of the unprivileged group.
        priv_name : str
            Name of the privileged group.

    """

    overall_weighted_diff = 0.0
    total_count = 0

    # We assume bins are aligned by index
    for b_unpriv, b_priv in zip(bins_unpriv, bins_priv):

        # Check bin ranges match
        if not (np.isclose(b_unpriv['lower'], b_priv['lower']) and
                np.isclose(b_unpriv['upper'], b_priv['upper'])):
            print("Warning: bin ranges don't match, skipping this pair")
            continue

        lower = b_unpriv['lower']
        upper = b_unpriv['upper']

        # Likelihood of positive class per group
        p_pos_unpriv = b_unpriv['likelihood_pos']
        p_pos_priv   = b_priv['likelihood_pos']

        # Absolute difference in likelihoods
        diff = abs(p_pos_unpriv - p_pos_priv)

        # Total samples in this bin across both groups
        bin_count = b_unpriv['count'] + b_priv['count']
        if bin_count == 0:
            continue

        # Weight by how many samples are in this bin
        overall_weighted_diff += diff * bin_count
        total_count += bin_count

        print(f"Bin [{lower:.2f}, {upper:.2f}]")
        print(f"  {unpriv_name}: count={b_unpriv['count']}, "
              f"P(Y=1|bin)={p_pos_unpriv:.3f}")
        print(f"  {priv_name}:   count={b_priv['count']}, "
              f"P(Y=1|bin)={p_pos_priv:.3f}")
        print(f"Difference in P(Y=1)| = {diff:.4f}\n")

    # Compute final weighted calibration gap across bins
    if total_count > 0:
        overall_group_calib_diff = overall_weighted_diff / total_count
        print(f"Overall (weighted) group calibration difference "
              f"(≈ ECE-style): {overall_group_calib_diff:.4f}")
    else:
        print("No overlapping bins with data to compare.")





