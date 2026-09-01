
#!/usr/bin/env python

"""
This script trains an Adversarial Debiasing model, in-processing, from the AIF360 library
on the COMPAS dataset to mitigate unfair predictions associated with
protected attributes (race).

It performs the following steps:
    - Loads and preprocesses the COMPAS dataset.
    - Converts the data into an AIF360 BinaryLabelDataset.
    - Splits the dataset into training, validation, and test sets.
    - Trains the Adversarial Debiasing model on the training set.
    - Evaluates and prints accuracy and fairness metrics on the test set.

"""

import tensorflow.compat.v1 as tf
# Required because AdversarialDebiasing uses TensorFlow 1.x graph mode
tf.disable_eager_execution()
from aif360.algorithms.inprocessing import AdversarialDebiasing
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

    # Split into train, validation, and test sets (validation is unused here).
    train, _, test = train_val_test_split(bld)

    # Initialise TensorFlow session
    sess = tf.Session()

    # Initialise the Adversarial Debiasing model.
    adv = AdversarialDebiasing(
        privileged_groups=PRIVILEGED_GROUPS,
        unprivileged_groups=UNPRIVILEGED_GROUPS,
        scope_name='adv_debias',
        debias=True,
        num_epochs=50,
        batch_size=128,
        sess=sess,
        seed=RANDOM_STATE
    )

    # Train the model.
    adv.fit(train)
    # Evaluate on the test set.
    adv_pred_test = adv.predict(test)

    # Print accuracy and fairness metrics.
    print_metrics(
        dataset_true=test,
        dataset_pred=adv_pred_test
    )

    # Close the TensorFlow session.
    sess.close()

if __name__ == "__main__":
    main()
