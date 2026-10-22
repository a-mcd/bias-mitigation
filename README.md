# Bias Mitigation

## Academic presentation

This project was developed as part of my Master’s studies. The accompanying
presentation covers the research background, fairness metrics, implementation,
experimental results, ethical considerations and conclusions.

[View the project presentation](docs/bias-mitigation-presentation.pdf)

## Project structure

```
data/        Cleaned COMPAS dataset
src/         Experiment scripts (one per mitigation approach)
src/biasmit/ Shared code (config, data loading, metrics)
analysis/    Exploratory analysis in R
results/     Generated figures and metric tables
docs/        Project presentation
```


## Setup

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip freeze | grep -iE "^(aif360|tensorflow|scikit-learn|pandas|numpy)=="
```


## COMPAS Dataset

The COMPAS dataset contains information used to assess the likelihood that an individual will reoffend within two years. It includes demographic attributes, criminal history, and a risk score derived from the COMPAS algorithm. Each row represents a single defendant, and the dataset is commonly used to study algorithmic fairness and racial bias. This is a cleaned and simplified version of the original ProPublica COMPAS dataset.
<br /><br />

- [ProPublica COMPAS data](data/propublica_data_for_fairml.csv)
- 6172 records.

| Column | Description |
|---|---|
| `Two_yr_Recidivism` | Target variable and ground truth (0 = did not reoffend, 1 = reoffended within two years) |
| `Number_of_Priors` | Total number of prior offenses |
| `score_factor` | Binary COMPAS risk prediction (0 = low risk, 1 = medium/high risk) |
| `Age_Above_FourtyFive` | 1 = age over 45, 0 = otherwise |
| `Age_Below_TwentyFive` | 1 = age under 25, 0 = otherwise |
| `African_American` | 1 = African American, 0 = not |
| `Asian` | 1 = Asian, 0 = not |
| `Hispanic` | 1 = Hispanic/Latino, 0 = not |
| `Native_American` | 1 = Native American, 0 = not |
| `Other` | 1 = not covered by the other race categories, 0 = not |
| `Female` | 1 = female, 0 = male |
| `Misdemeanor` | 1 = misdemeanor charge, 0 = felony |

<br />


## Exploratory Analysis (R)


| Script | Description |
|---|---|
| `analyse_data.R` | Produces 4 plots exploring potential disparities: <br /> - Representation by race and gender <br /> - Prior convictions distribution across racial groups <br /> - Recidivism outcomes split by race and sex <br /> - Recidivism outcomes by age |
| `baseline.py` | - Performs Principal Component Analysis (PCA) to explore structure in predictor variables. <br /> - Visualises how individuals distribute across PCs and whether structure aligns with race <br /> - Identifies which variables contribute most to separation in the PCA space <br /> |


<br />

## Training and Evaluation (Python)

### Shared code (`src/biasmit/`)

- `fairness_config.py` This is the fairness configuration used across all python scripts. This file defines a single source of truth for protected and privileged attributes.
- `data.py` The data utilities module provides a set of helper functions used in COMPAS fairness experiments, including functions to load raw data, convert to AIF360 BinaryLabelDataset, split into train/val/test sets.
- `metrics_utils.py` The metrics utilities module provides a function to evaluate machine-learning models using standard fairness metrics.

Metrics from AIF360:
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

### Experiments (`src/`)

Run each script from the repository root, e.g. `python src/baseline.py`.

| Script | Description |
|---|---|
| `evaluate_compas.py` | Fairness audit of the original COMPAS `score_factor` using AIF360 metrics. No model is trained. |
| `baseline.py` | Logistic regression on the cleaned data, used as the comparison point. |
| `dir_only.py` | Pre-processing with Disparate Impact Remover (DIR), then logistic regression. |
| `adv_only.py` | In-processing with AIF360 Adversarial Debiasing to reduce unfair predictions associated with race. |
| `eqodds_only.py` | Logistic regression followed by Equalized Odds post-processing. |
| `dir_adv_eqodds.py` | Three-stage pipeline: Pre-processing-DIR, then In-processing Adversarial Debiasing, then Post-processing Equalized Odds. |

