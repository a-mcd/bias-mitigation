# Bias Mitigation

## Academic presentation

This project was developed as part of my Master’s studies. The accompanying
presentation covers the research background, fairness metrics, implementation,
experimental results, ethical considerations and conclusions.

[View the project presentation](bias-mitigation-presentation.pdf)

## COMPAS Dataset

The COMPAS dataset contains information used to assess the likelihood that an individual will reoffend within two years. It includes demographic attributes, criminal history, and a risk score derived from the COMPAS algorithm. Each row represents a single defendant, and the dataset is commonly used to study algorithmic fairness and racial bias.This ia a cleaned and simplified version of the original ProPublica COMPAS dataset.
<br />
6172 records.

Key Columns
- Two_yr_Recidivism: Target variable and ground-truth. (0 = did not reoffend, 1 = reoffended within two years).
- Number_of_Priors: Total number of prior offenses.
- score_factor: Binary COMPAS risk prediction (0 = low risk, 1 = medium/high risk).
- Age_Above_FourtyFive: Age indicators used in COMPAS scoring (1 = age > 45, 0 = otherwise).
- Age_Below_TwentyFive: Age indicators used in COMPAS scoring (1 = age > 45, 0 = otherwise).
- African_American: (1 = African American, 0 = not).
- Asian: (1 = Asian, 0 = not).
- Hispanic: (1 = Hispanic/Latino, 0 = not).
- Native_American: (1 = Native American, 0 = not).
- Other: (1= not covered by other categories, 0 = not).
- Female: (1 = female, 0 = male).
- Misdemeanor: (1 = misdemeanor charge, 0 = felony).

<br />

## Pip Requirements

- pip install "aif360[all]"
- pip install tensorflow
- pip install scikit-learn
- pip install pandas
- pip install numpy

## COMPAS Dataset Analysis (R)

### AnalyseData.R
Produces 4 plots exploring potential disparities:
- Representation by race and gender
- Prior convictions distribution across racial groups
- Recidivism outcomes split by race and sex
- Recidivism outcomes by age

<br />

### PCA.R
- Performs Principal Component Analysis (PCA) to explore structure in predictor variables. 
- Visualises how individuals distribute across PCs and whether structure aligns with race
- Identifies which variables contribute most to separation in the PCA space

<br />

## Training and Evaluation (Python)

### fairness_config.py

This is the fairness configuration used across the all python scripts. This file defines a single source of truth for protected and privileged attributes.

<br />

### data_utils.py

Data utilities provides a set of helper fucntions used in COMPAS fairness experiments.

Includes functions to 
    - load raw data
    - convert to AIF360 BinaryLabelDataset,
    - split into train/val/test sets.

<br />

### metrics_utils.py

Metrics utils provides a fucntion to evaluate machine-learning models using standard fairness metrics.

Metrivs from AIF360:
- Accuracy, 
- Demographic parity
- Disparate impact
- Equalized odds: TPR/FPR differences
- Equality of opportunity
- Predictive parity: NPV and PPV 

Custom score metrics:
- Balance for the Positive Class
- Balance for the Negative Class
- Group calibration comparison similar to ECE-style weighting

<br/>

### evaluate_compas_fairness.py

python evaluate_compas_fairness.py
<br /><br />
This script evaluates the fairness of the original COMPAS risk score_factor
using the AIF360 fairness metrics framework. It does not train a model, it performs a fairness audit of the
existing COMPAS scoring system.

<br />

### baseline.py

python baseline.py
<br /><br />
This script trains a simple logistic regression model on the cleaned COMPAS dataset.
The purpose of this baseline model is to provide a comparison point for fairness
analysis.

<br />

### dir_only.py

python dir_only.py
<br /><br />
This script applies the Disparate Impact Remover (DIR) pre-processing
technique to the COMPAS dataset to mitigate bias before training a
logistic regression classifier. The performance and fairness metrics
are then evaluated on the test set.

<br />

### adv_only.py

python adv_only.py
<br /><br />
This script trains an Adversarial Debiasing model, in-processing bias mitigation, from the AIF360 library
on the COMPAS dataset to mitigate unfair predictions associated with
protected attributes (race).

<br />

### eqodds_only.py

python eqodds_only.py
<br /><br />
This script trains a Logistic Regression classifier on the COMPAS dataset,
then applies Equalized Odds post-processing from AIF360 to correct disparities
across protected groups.

<br />

### dir_adv_eqodds.py

python dir_adv_eqodds.py
<br /><br />
This script applies a three-stage fairness mitigation approach to the COMPAS dataset:
1. Pre-processing with Disparate Impact Remover (DIR).
2. In-processing with Adversarial Debiasing.
3. Post-processing with Equalized Odds (EO) adjustment.
