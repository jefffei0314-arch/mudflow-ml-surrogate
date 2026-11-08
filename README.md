
# Machine Learning Surrogate Modelling of Mudflow Depth Profiles

## 1. Project Overview

This project investigates the application of machine learning regression models to predict mudflow depth profiles under varying yield-stress conditions.

The numerical simulation data originate from a depth-averaged mudflow model using Quadratic rheology, developed in previous research by Tan et al. (2025) [1]..

Two machine learning methods are compared:

- **Polynomial Ridge Regression:** Polynomial features with L2 regularization.
- **Random Forest Regression:** An ensemble of decision trees for modelling nonlinear relationships.

Hyperparameters for both models are selected using grid search with group-aware cross-validation.

The objective is to evaluate how accurately these models predict mudflow depth profiles at yield stresses excluded from model training.

## 2. Dataset

The dataset contains nine yield-stress conditions:

**20, 50, 80, 200, 500, 800, 1000, 2000, and 3000 Pa.**

Each condition contains 35,000 spatial observations, giving a total of **315,000 observations**.

### Input Features

1. Normalized spatial coordinate, `x/x0`
2. Yield stress, `τy` (Pa)

### Prediction Target

- Normalized flow depth, `h/h0`

Normalization constants:

- `x0 = 305 m`
- `h0 = 30.5 m`

The main dataset is:

`quadratic_yieldstress_combined_9_cases.csv`

Three additional reference datasets are provided:

- `quadratic_50_clean.csv`
- `quadratic_500_clean.csv`
- `quadratic_2000_clean.csv`

These reference datasets are used to verify and evaluate the held-out predictions.

## 3. Machine Learning Methodology

### 3.1 Training and Testing

Three independent held-out yield-stress experiments are conducted at:

- 50 Pa
- 500 Pa
- 2000 Pa

For each experiment, the complete target yield-stress profile is excluded from training.

The remaining eight profiles (280,000 observations) are used for model training and validation, while the excluded profile (35,000 observations) is reserved for final testing.

This prevents observations from the same held-out yield-stress profile from appearing in both training and testing.

### 3.2 Hyperparameter Validation

Hyperparameters are selected using `GridSearchCV` with eight-fold `GroupKFold` cross-validation.

The grouping variable is yield stress, ensuring that each validation fold contains a complete yield-stress profile excluded from the corresponding training fold.

The following hyperparameters are evaluated:

**Polynomial Ridge Regression**

- Polynomial degree: 2, 3, 4
- Regularization parameter (alpha): 0.01, 0.1, 1, 10, 100

**Random Forest Regression**

- Number of trees: 100, 200
- Maximum tree depth: 8, 12
- Minimum samples per leaf: 1, 3

The optimal configuration is selected by minimizing the mean cross-validation mean squared error (MSE).

The selected model is then refitted using all eight training profiles before final testing.

### 3.3 Selected Hyperparameters

| Held-out stress | Ridge degree | Ridge alpha | RF trees | RF max depth | RF min leaf |
|---|---:|---:|---:|---:|---:|
| 50 Pa | 4 | 1 | 200 | 12 | 1 |
| 500 Pa | 4 | 10 | 200 | 12 | 1 |
| 2000 Pa | 3 | 100 | 100 | 12 | 1 |

Negative predicted depths are clipped to zero.

Model performance is evaluated using:

- Coefficient of determination (R²)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

RMSE and MAE are reported in metres.

## 4. Installation

### Requirements

- Python 3
- NumPy
- pandas
- Matplotlib
- scikit-learn

Install the required packages:

```bash
pip install -r requirements.txt
```

### Required Files

The main Python script and numerical datasets must be placed in the same directory.

Recommended repository structure:

```text
CV7001-Assignment-3/
|
|-- CV7001_Assignment_3.py
|-- ML_Inference_Demo.py
|-- README.md
|-- requirements.txt
|
|-- quadratic_yieldstress_combined_9_cases.csv
|-- quadratic_50_clean.csv
|-- quadratic_500_clean.csv
|-- quadratic_2000_clean.csv
|
|-- ML_Results/
    |-- model_performance_summary.csv
    |-- comparison_yieldstress_50.csv
    |-- comparison_yieldstress_500.csv
    |-- comparison_yieldstress_2000.csv
    |-- flow_profile_yieldstress_50.png
    |-- flow_profile_yieldstress_500.png
    |-- flow_profile_yieldstress_2000.png
    |-- prediction_error_yieldstress_50.png
    |-- prediction_error_yieldstress_500.png
    |-- prediction_error_yieldstress_2000.png
    |
    |-- Validation/
        |-- validation_summary.csv
```

The result files are generated automatically by the main Python script. The full validation diagnostics are also saved under `ML_Results/Validation/`.

## 5. Running the Experiments

Run the main training, validation, and testing algorithm:

```bash
python CV7001_Assignment_3.py
```

The program automatically:

1. Loads the numerical simulation dataset.
2. Separately excludes the 50, 500, and 2000 Pa profiles.
3. Performs eight-fold group-aware cross-validation.
4. Selects the optimal hyperparameters for both models.
5. Refits the selected models using the remaining eight profiles.
6. Evaluates predictions against the held-out numerical profiles.
7. Calculates R², RMSE, and MAE.
8. Saves prediction comparisons, validation results, and figures.

The main outputs are stored in `ML_Results/`.

The validation results, including the selected hyperparameters, are stored in `ML_Results/Validation/`.

The full grid search is performed whenever the main script is executed.

## 6. Inference Demonstration

The inference demonstration predicts mudflow depth at a user-specified spatial coordinate using previously validated hyperparameters.

Run:

```bash
python ML_Inference_Demo.py --stress 500 --x 610
```

Where:

- `--stress 500` specifies the yield stress in Pa.
- `--x 610` specifies the spatial coordinate in metres.

The demonstration supports the three validated held-out yield-stress cases: **50, 500, and 2000 Pa**.

For each selected case, the corresponding yield-stress profile is excluded from training.

The script reads the selected hyperparameters from:

`ML_Results/Validation/validation_summary.csv`

It then trains both models using those configurations and performs inference without repeating the grid search.

### Example: 500 Pa at x = 610 m

| Model | Normalized depth (h/h0) | Predicted depth (m) |
|---|---:|---:|
| Polynomial Ridge | 0.122957 | 3.750178 |
| Random Forest | 0.103177 | 3.146887 |

These predictions demonstrate inference at a specified location within a held-out yield-stress profile.

They should not be interpreted as independent evidence of prediction accuracy without comparison against the numerical reference.

**Note:** If `validation_summary.csv` is unavailable, run the main training script first to generate it.

## 7. Results

### 7.1 Held-Out Testing Performance

| Yield stress | Model | R² | RMSE (m) | MAE (m) |
|---|---|---:|---:|---:|
| 50 Pa | Polynomial Ridge | 0.7657 | 0.4315 | 0.3408 |
| 50 Pa | Random Forest | 0.9711 | 0.1514 | 0.0440 |
| 500 Pa | Polynomial Ridge | 0.9024 | 0.4569 | 0.3044 |
| 500 Pa | Random Forest | 0.7053 | 0.7942 | 0.4382 |
| 2000 Pa | Polynomial Ridge | 0.8917 | 0.8915 | 0.7171 |
| 2000 Pa | Random Forest | 0.7454 | 1.3665 | 0.9409 |

Random Forest achieved better performance at 50 Pa, whereas Polynomial Ridge obtained lower prediction errors at 500 and 2000 Pa.

However, the predicted flow-depth profiles reveal differences that are not fully represented by the statistical metrics.

### 7.2 Flow-Depth Profile Comparison

Random Forest reproduced the characteristic mudflow profile shape more closely, whereas Polynomial Ridge produced overly smooth profiles.

Random Forest also tended to predict positive depths beyond the numerical flow front, particularly at higher yield stresses.

The numerical model includes a yield-stress resistance formulation that influences flow propagation, but neither ML model explicitly incorporates the governing equations.

Therefore, lower overall prediction errors do not necessarily indicate better reproduction of the physical mudflow profile.

### 7.3 Figures

The repository includes six figures:

**Flow-depth profile comparisons**

- `flow_profile_yieldstress_50.png`
- `flow_profile_yieldstress_500.png`
- `flow_profile_yieldstress_2000.png`

**Signed prediction errors**

- `prediction_error_yieldstress_50.png`
- `prediction_error_yieldstress_500.png`
- `prediction_error_yieldstress_2000.png`

These figures compare the ML predictions against the numerical simulations at 180 s.

## 8. Limitations

Although the dataset contains 315,000 spatial observations, they represent only nine independent yield-stress conditions.

The evaluation is limited to three held-out yield stresses within the range represented by the training data. Generalization beyond this range has not been established.

Neither ML model explicitly enforces the governing flow equations, rheological resistance formulation, or mass conservation.

Polynomial Ridge produces overly smooth profiles, while Random Forest may overestimate the flow-front position.

These limitations are important because errors in runout distance can affect the reliability of mudflow hazard and inundation assessments.

## 9. Reproducibility

The experiments use the following settings:

- ML library: scikit-learn
- Cross-validation: eight-fold GroupKFold
- Hyperparameter optimization: GridSearchCV
- Model selection criterion: minimum mean validation MSE
- Random Forest random seed: `random_state=42`
- Training profiles: eight yield-stress cases per experiment
- Testing profile: one completely excluded yield-stress case
- Normalization constants: `x0 = 305 m`, `h0 = 30.5 m`

Polynomial Ridge uses input standardization, polynomial feature expansion, a second standardization step, and L2-regularized regression.

Random Forest uses the selected tree configurations with a fixed random seed.

The code and datasets are provided so that the experiments can be reproduced.

## 10. Reference

Tan, J. F., Tion, P. H., and Lau, S. C. (2025). *Parametric Sensitivity Analyses of Mudflow Rheology with Depth-Averaged Model*. Journal of Advanced Research in Fluid Mechanics and Thermal Sciences, 131(2), 78–94.

DOI: https://doi.org/10.37934/arfmts.131.2.7894

The numerical simulation datasets used in this project originate from this research.

The machine learning implementation uses the scikit-learn library.

## 11. AI Assistance Disclosure

ChatGPT (OpenAI) assisted with Python implementation, debugging and LaTeX formatting.

The author provided the numerical simulation data, selected the research problem, ran the experiments, reviewed the results, drafted the report and interpreted their engineering significance.

## 12. Conclusion and Reflection

The main contribution of this project is the application and evaluation of machine learning surrogate models using numerical mudflow simulation data from previous research.

The results demonstrate the potential of ML to approximate flow-depth profiles under unseen yield-stress conditions, which could reduce the computational effort required for parametric studies.

In civil engineering, such models could support rapid sensitivity analysis and preliminary mudflow hazard assessment.

However, the difficulty in accurately predicting sharp flow fronts highlights an important limitation, as errors in flow extent may affect inundation assessment.

Future work should focus on improving physical consistency, expanding the training dataset, and improving flow-front prediction accuracy before these models can be applied reliably in engineering practice.
