# Causal Inference: Does Physical Activity Reduce Diabetes Risk?

A causal inference analysis investigating whether increasing physical activity **directly causes** a reduction in diabetes risk, or whether this association is entirely driven by confounding variables.

---

## Table of Contents

- [Research Question](#research-question)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [1. Causal DAG](#1-causal-dag)
  - [2. Identification](#2-identification)
  - [3. Data Mapping](#3-data-mapping)
  - [4. Estimation](#4-estimation)
- [Results](#results)
- [Conclusion](#conclusion)
- [Tech Stack](#tech-stack)

---

## Research Question

> **Does increasing physical activity directly cause a reduction in diabetes risk, or is this effect entirely attributable to confounders?**

This is a causal question — not merely a correlational one. Standard regression or observational analysis cannot answer it without explicitly accounting for the web of confounding variables that influence both physical activity and diabetes risk. This project applies a formal causal inference pipeline to answer it rigorously.

---

## Dataset

**CDC Diabetes Health Indicators Dataset** (BRFSS 2015)

| Property | Details |
|---|---|
| Source | CDC Behavioral Risk Factor Surveillance System (BRFSS), 2015 |
| Features | 22 variables (demographics, lab results, lifestyle survey responses) |
| Target Variable | `Diabetes_012` — 0 = Healthy, 1 = Pre-diabetic, 2 = Diabetic |
| Treatment Variable | `PhysActivity` — Binary (1 = physically active, 0 = not) |

Key variables used in this analysis:

- **Exposure:** `PhysActivity` (binary)
- **Outcome:** `Diabetes_012` (multiclass)
- **Confounders:** `Age`, `Sex`, `Income`, `Education`

---

## Methodology

This project follows the standard four-step causal inference pipeline:

### 1. Causal DAG

A Directed Acyclic Graph (DAG) was constructed to encode domain knowledge about the causal relationships between variables. The DAG was built using the **DoWhy** library.

Key relationships encoded in the DAG:

- `Age`, `Sex`, `Income`, `Education` → `PhysActivity` (confounders)
- `Age`, `Sex`, `Income`, `Ethnicity` → `Diabetes_Risk` (direct effects)
- `PhysActivity` → `BMI` → `Diabetes_Risk` (mediation path)
- `PhysActivity` → `Cholesterol` → `Diabetes_Risk` (mediation path)
- `PhysActivity` → `BP` → `Diabetes_Risk` (mediation path)
- `Mental Health`, `Difficulty Walking` → `PhysActivity`
- `Mental Health` also serves as a candidate **Instrumental Variable (IV)**

> Note: 4 variables (`Genetics`, `Ethnicity`, `Diet`, `Mental Health`) were assumed unobserved as they lacked direct measures in the dataset.

### 2. Identification

Using the DAG, DoWhy identified the **minimal deconfounding set** (backdoor adjustment set) required to isolate the causal effect of physical activity on diabetes risk:

```
d/d[PhysActivity] E[Diabetes_012 | Sex, Age, Education, Income]
```

**Interpretation:** To find the true causal effect, we compare people with different physical activity levels while holding `Age`, `Sex`, `Income`, and `Education` constant. Any remaining difference in diabetes risk can then be attributed to physical activity itself.

The identification step also found:
- A valid **Instrumental Variable** estimand using `Mental Health` as an instrument
- No valid **Front-door** estimand (no unconfounded mediator chain found)

### 3. Data Mapping

The CDC dataset was mapped to the theoretical DAG variables:

| DAG Variable | CDC Dataset Column |
|---|---|
| `PhysActivity` | `PhysActivity` |
| `Diabetes_Risk` | `Diabetes_012` |
| `Age` | `Age` |
| `Sex` | `Sex` |
| `Income` | `Income` |
| `Education` | `Education` |
| `BP` | `HighBP` |
| `Cholesterol` | `HighChol` |
| `BMI` | `BMI` |
| `Difficulty Walking` | `DiffWalk` |
| `Diet` (proxy) | `Veggies` |
| `Mental Health` | `MentHlth` |

After verifying that adequate measures existed for all variables in the deconfounding set, the re-specified DAG was passed back into DoWhy for estimation.

### 4. Estimation

The causal effect was estimated using the **backdoor linear regression** method via DoWhy, and cross-validated with **Statsmodels OLS**:

```python
# DoWhy estimation
deconfounding_estimate = deconfounding_model.estimate_effect(
    deconfounding_estimands,
    method_name="backdoor.linear_regression",
    confidence_intervals=True,
    test_significance=True
)

# Statsmodels cross-validation
reg_model = smf.ols(
    formula='Diabetes_012 ~ PhysActivity + Age + Sex + Income + Education',
    data=df
)
```

---

## Results

| Metric | Value |
|---|---|
| **ATE (Average Treatment Effect)** | **−0.1230** |
| p-value | 0.000 (statistically significant) |
| 95% Confidence Interval | [−0.1293, −0.1167] |

Both DoWhy and Statsmodels produced **identical results**, providing strong convergent validity.

**Interpretation:**

> Increasing the treatment variable `PhysActivity` from 0 to 1 causes an **average decrease of 0.123 points** in the expected diabetes risk score (`Diabetes_012`), after controlling for Age, Sex, Income, and Education.

The **negative effect** means physical activity pulls individuals toward 0 (healthy), which is consistent with clinical expectations and confirms the model is behaving correctly.

---

## Conclusion

Physical activity has a **statistically significant, direct causal effect** on reducing diabetes risk — this effect is **not** entirely attributable to confounders. After adjusting for age, sex, income, and education using the backdoor criterion:

- Physically active individuals have a diabetes risk score approximately **0.12 points lower** than inactive individuals, on average across the population.
- The result is highly significant (p ≈ 0) with a tight 95% confidence interval.
- The direction of the effect aligns with established clinical knowledge, lending further credibility to the causal model.

This analysis demonstrates that interventions encouraging physical activity could be expected to produce a genuine reduction in diabetes risk at the population level.

---

## Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `dowhy` | Causal model construction, identification, and estimation |
| `statsmodels` | OLS regression for cross-validation |
| `matplotlib` | DAG and distribution visualisation |

---

## How to Run

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install dowhy statsmodels pandas numpy matplotlib
   ```
3. Download the [CDC Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) and update the file path in the notebook
4. Run `Causal_Inference_Project1.ipynb` from top to bottom
