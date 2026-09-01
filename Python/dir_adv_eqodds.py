#!/usr/bin/env python

"""
This script applies a three-stage fairness mitigation approach to the COMPAS dataset:
1) Pre-processing with Disparate Impact Remover (DIR).
2) In-processing with Adversarial Debiasing.
3) Post-processing with Equalized Odds (EO) adjustment.

Fairness metrics are evaluated on the final test predictions.
"""

import tensorflow.compat.v1 as tf
# Required because AdversarialDebiasing uses TensorFlow 1.x graph mode
tf.disable_eager_execution()

from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.algorithms.inprocessing import AdversarialDebiasing
from aif360.algorithms.postprocessing import EqOddsPostprocessing

from common.data_utils import (
    load_raw_compas, to_binary_label_dataset, train_val_test_split,
)
from config.fairness_config import (
    PRIVILEGED_GROUPS, UNPRIVILEGED_GROUPS,
    RANDOM_STATE
)
from common.metrics_utils import print_metrics

def main():

    # Load and prepare the COMPAS dataset.
    df = load_raw_compas()
    bld = to_binary_label_dataset(df)
    train, val, test = train_val_test_split(bld)

    # 1) Pre-processing: DIR on train
    dir_ = DisparateImpactRemover(repair_level=1.0)
    train_repaired = dir_.fit_transform(train)

    # 2) In-processing: Adversarial Debiasing on repaired train
    sess = tf.Session()
    adv = AdversarialDebiasing(
        privileged_groups=PRIVILEGED_GROUPS,
        unprivileged_groups=UNPRIVILEGED_GROUPS,
        scope_name='adv_dir',
        debias=True,
        num_epochs=50,
        batch_size=128,
        sess=sess,
        seed=RANDOM_STATE
    )
    adv.fit(train_repaired)

    # Get predictions on val + test (unrepaired, to evaluate on original data)
    adv_val_pred = adv.predict(val)
    adv_test_pred = adv.predict(test)

    # 3) Post-processing: Equalized Odds fitted on val
    eq = EqOddsPostprocessing(
        privileged_groups=PRIVILEGED_GROUPS,
        unprivileged_groups=UNPRIVILEGED_GROUPS,
        seed=RANDOM_STATE
    )
    eq = eq.fit(val, adv_val_pred)

    # Apply the EO adjustments to test predictions
    final_test_pred = eq.predict(adv_test_pred)

    # Evaluate fairness metrics for this model.
    print_metrics(
        dataset_true=test,
        dataset_pred=final_test_pred
    )

    # Close the TensorFlow session.
    sess.close()

if __name__ == "__main__":
    main()
