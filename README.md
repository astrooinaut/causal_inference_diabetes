# Advanced Causal Inference: Physical Activity & Diabetes Risk

An extension of a foundational causal inference analysis, this project applies progressively advanced methods to answer the same research question using the CDC Diabetes Health Indicators Dataset. Starting from a simple backdoor adjustment, the pipeline advances through propensity score matching and inverse probability of treatment weighting (IPTW), providing multiple independent lines of causal evidence that converge on the same conclusion.
______________________________________________________________
## Table of Contents

- [Research Question](#research-question)
- [Dataset](#dataset)
- [Differences from Project 1](#differences-from-project-1)
- [Methods Pipeline](#methods-pipeline)
  - [1. Causal DAG & Identification](#1-causal-dag--identification)
  - [2. Simple Backdoor Estimation (Baseline)](#2-simple-backdoor-estimation-baseline)
  - [3. Propensity Score Estimation](#3-propensity-score-estimation)
  - [4. Propensity Score Matching (PSM)](#4-propensity-score-matching-psm)
  - [5. Inverse Probability of Treatment Weighting (IPTW)](#5-inverse-probability-of-treatment-weighting-iptw)
- [Results](#results)
- [Conclusion](#conclusion)
- [Tech Stack](#tech-stack)
- [How to Run](#how-to-run)

## Research Question

> **Does increasing physical activity directly cause a reduction in diabetes risk, or is this effect entirely attributable to confounders?**

This project builds directly on [Project 1](../causal_inference_project1/) by extending the single-method backdoor regression approach into a multi-method causal pipeline, with each method providing an independent robustness check on the causal estimate.
______________________________________________________________
## Dataset

**CDC Diabetes Health Indicators Dataset** (BRFSS 2015)

| Property | Details |
|---|---|
| Source | CDC Behavioral Risk Factor Surveillance System (BRFSS), 2015 |
| Samples | 253,680 |
| Features | 22 variables (demographics, lab results, lifestyle survey) |
| Treatment | `PhysActivity` — Binary (1 = active, 0 = inactive) |
| Outcome | `Diabetes_012` — recoded to binary (0 = healthy, 1 = pre-diabetic or diabetic) |

**Key change from Project 1:** The outcome variable was recoded from 3 classes (healthy / pre-diabetic / diabetic) to **2 classes** (healthy vs. any diabetes). Pre-diabetic cases (class 1) were removed and diabetic cases (class 2) were recoded to 1, enabling logistic regression and odds ratio interpretation in the IPTW stage.
______________________________________________________________
## Differences from Project 1

| Aspect | Project 1 | Project 2 |
|---|---|---|
| Outcome variable | 3-class (0, 1, 2) | Binary (0 = healthy, 1 = diabetic/pre-diabetic) |
| Estimation methods | Backdoor linear regression only | Backdoor + PSM + IPTW |
| Confounder adjustment | OLS regression | Propensity score model, matching, weighting |
| Balance verification | None | SMD plots before & after matching |
| Effect measure | Average Treatment Effect (ATE, linear) | ATE (linear) + Odds Ratios (logistic) |
| Sample size | ~253k (subset used) | Full 253,680 used |
| Robustness checks | Statsmodels cross-validation | Three independent methods converge |
______________________________________________________________
## Methods Pipeline

### 1. Causal DAG & Identification

The same DAG from Project 1 was retained, encoding domain knowledge about the causal relationships between physical activity, diabetes risk, and confounders including age, sex, income, education, BMI, blood pressure, cholesterol, genetics, and mental health.

DoWhy's `identify_effect()` was used to derive the **minimal deconfounding set** via the backdoor criterion:

```
d/d[PhysActivity] E[Diabetes_012 | Income, Sex, Education, Age]
```

**Interpretation:** Conditioning on Age, Sex, Income, and Education blocks all backdoor paths from physical activity to diabetes risk, isolating the direct causal effect.

### 2. Simple Backdoor Estimation (Baseline)

A linear regression was fitted via both DoWhy and Statsmodels as a baseline, replicating the Project 1 approach on the recoded binary outcome:

```python
Diabetes_012 ~ PhysActivity + Age + Sex + Income + Education
```

Both implementations produced identical results, confirming the estimate is robust to the choice of library.

| Metric | Value |
|---|---|
| ATE | **−0.0633** |
| p-value | 0.000 |
| 95% CI | [−0.0666, −0.0600] |

**Interpretation:** Being physically active reduces the probability of a diabetes diagnosis by approximately **6.3 percentage points** on average, after adjusting for age, sex, income, and education.

### 3. Propensity Score Estimation

A **propensity score** is the estimated probability that a given individual is physically active, given their observed characteristics. It compresses all confounders into a single scalar that can be used for matching or weighting.

A logistic regression model was fitted to estimate propensity scores from four confounders (Age, Sex, Income, Education):

```python
P(PhysActivity = 1 | Age, Sex, Income, Education)
```

A **common support check** (overlap plot) confirmed that the propensity score distributions of the treated (active) and control (inactive) groups substantially overlap — a necessary condition for valid matching and weighting. Neither group was exclusively clustered near 0 or 1.

### 4. Propensity Score Matching (PSM)

PSM creates a balanced pseudo-experiment by pairing each physically active individual with an inactive individual who has a nearly identical propensity score — i.e., who looks essentially the same in age, sex, income, and education.

**Procedure:**

1. Treated group was subsampled to match the size of the control pool (61,760 each)
2. A `cKDTree` (k-d tree nearest-neighbour search) was used for fast 1:1 nearest-neighbour matching on propensity score
3. **Standardised Mean Difference (SMD)** was calculated before and after matching for all four confounders to verify balance

**Balance results:**

| Confounder | SMD Before Matching | SMD After Matching |
|---|---|---|
| Age | 0.219 | 0.000016 |
| Education | 0.464 | 0.000086 |
| Income | 0.459 | 0.000025 |
| Sex | 0.076 | 0.000065 |

All SMDs after matching are effectively zero (well below the 0.1 threshold), confirming that the matched groups are highly comparable. Before matching, active people tended to be older, more educated, and higher-income — all factors correlated with diabetes risk — which would have severely confounded a naïve comparison.

### 5. Inverse Probability of Treatment Weighting (IPTW)

IPTW is an alternative to matching that reweights the full dataset rather than discarding unmatched observations. Each individual is up-weighted if they belong to the group they were unlikely to be in (given their characteristics), creating a **pseudo-population** in which treatment is independent of confounders.

**Procedure:**

1. Propensity scores were estimated using a richer set of 12 risk factors: Age, Sex, Income, Education, HighBP, HighChol, BMI, Smoker, Veggies, HvyAlcoholConsump, MentHlth, DiffWalk
2. **Stabilised IPTW weights** were computed to reduce variance from extreme propensity scores:
   - Treated: `weight = P(Treatment=1) / propensity_score`
   - Control: `weight = P(Treatment=0) / (1 - propensity_score)`
3. Weights were **trimmed at the 1st and 99th percentiles** to prevent extreme values from dominating the analysis
4. A **weighted logistic regression** (GLM with Binomial family) was fitted on the IPTW-weighted data, modelling the full set of risk factors alongside physical activity
5. **Odds Ratios (OR)** with 95% CIs and significance levels were extracted for all variables and visualised as a **forest plot**

**Propensity score summary post-estimation:**

| Statistic | Value |
|---|---|
| Mean | 0.757 |
| Std | 0.148 |
| Min | 0.051 |
| Max | 0.938 |

**IPTW weight summary after stabilisation and trimming:**

| Statistic | Value |
|---|---|
| Mean | 0.994 |
| Std | 0.337 |
| Min | 0.354 |
| Max | 2.419 |
______________________________________________________________
## Results

### Effect of Physical Activity on Diabetes Risk Across Methods

| Method | Estimate | 95% CI | p-value |
|---|---|---|---|
| Backdoor Linear Regression (DoWhy) | ATE = −0.0633 | [−0.0666, −0.0600] | < 0.001 |
| Backdoor Linear Regression (Statsmodels) | ATE = −0.0633 | [−0.0666, −0.0600] | < 0.001 |
| IPTW Weighted Logistic Regression | OR = **0.8496** | [0.827, 0.873] | < 0.001 |

All three methods agree: **physical activity is significantly protective against diabetes**, and this effect is not explained away by confounders.

The IPTW odds ratio of **0.85** means that physically active individuals have **15% lower odds** of being diabetic or pre-diabetic compared to inactive individuals with the same risk profile.

### IPTW Full Results — Odds Ratios for All Risk Factors

| Variable | OR | 95% CI | Significance |
|---|---|---|---|
| **PhysActivity** | **0.850** | **[0.827, 0.873]** | *** |
| HighBP | 2.382 | [2.319, 2.446] | *** |
| HighChol | 1.991 | [1.942, 2.041] | *** |
| DiffWalk | 1.589 | [1.545, 1.635] | *** |
| Age | 1.135 | [1.129, 1.140] | *** |
| Sex | 1.330 | [1.298, 1.363] | *** |
| BMI | 1.063 | [1.062, 1.065] | *** |
| Smoker | 1.042 | [1.017, 1.068] | *** |
| MentHlth | 1.009 | [1.007, 1.010] | *** |
| HvyAlcoholConsump | 0.489 | [0.457, 0.523] | *** |
| Veggies | 0.929 | [0.903, 0.955] | *** |
| Income | 0.910 | [0.905, 0.916] | *** |
| Education | 0.914 | [0.903, 0.926] | *** |

**** p < 0.001*

Results were visualised as a **forest plot** with OR point estimates, confidence intervals, and a reference line at OR = 1 (no effect), with shading distinguishing protective (OR < 1) from harmful (OR > 1) factors.
______________________________________________________________
## Conclusion

Across three independent estimation strategies — backdoor linear regression, propensity score matching, and IPTW weighted logistic regression — the causal evidence consistently shows that **physical activity has a direct, statistically significant protective effect on diabetes risk**, independent of age, sex, income, education, and a broad range of clinical risk factors.

Key findings:

- The **ATE of −0.063** (backdoor) indicates active individuals have a ~6.3 percentage point lower probability of being diabetic
- The **OR of 0.85** (IPTW) indicates active individuals have 15% lower odds of diabetes, even after adjusting for 12 confounders
- **Propensity score matching** confirmed that naïve comparisons were highly confounded — active people were substantially older, more educated, and higher-income before matching. After matching, all SMDs dropped to near zero, isolating the causal effect
- High blood pressure (OR = 2.38) and high cholesterol (OR = 1.99) were the strongest risk-increasing factors in the IPTW model
- Heavy alcohol consumption (OR = 0.49) and vegetable consumption (OR = 0.93) were independently protective, consistent with clinical knowledge

The convergence of three methodologically distinct approaches strengthens the causal claim considerably beyond what any single method alone could provide.
______________________________________________________________
## Tech Stack

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `dowhy` | Causal model, DAG specification, identification, estimation |
| `statsmodels` | OLS cross-validation, weighted GLM (IPTW) |
| `scikit-learn` | Logistic regression for propensity score estimation, StandardScaler |
| `scipy` | `cKDTree` nearest-neighbour matching for PSM |
| `matplotlib` | Overlap plots, SMD balance plots, forest plot |
______________________________________________________________
## How to Run

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install dowhy statsmodels scikit-learn scipy pandas numpy matplotlib
   ```
3. Download the [CDC Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) and update the file path in the notebook
4. Run `Advanced_Causal_Inference_Project2.ipynb` from top to bottom

> **Note:** The notebook builds sequentially — propensity scores computed in the matching section are reused in the IPTW section, so cells must be run in order.
______________________________________________________________
## Related Projects
- **[Project 1 — Simple Causal Inference](../causal_inference_project1/):** The foundational backdoor adjustment approach on the 3-class outcome using the same DAG and dataset.
