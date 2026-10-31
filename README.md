# Machine Learning Surrogate Modelling of Mudflow Depth Profiles

## 1. Project Overview

This project investigates the application of machine learning regression models to predict mudflow depth profiles under varying yield-stress conditions.

The numerical simulation data originate from a depth-averaged mudflow model using quadratic rheology. Two machine learning methods are compared:

- Polynomial Ridge Regression (degree 4, regularization parameter α = 1.0)
- Random Forest Regression (200 trees, maximum depth 12)

The objective is to evaluate how accurately these models can predict flow-depth profiles at yield stresses excluded from model training.

## 2. Dataset

The dataset contains nine yield-stress conditions: 20, 50, 80, 200, 500, 800, 1000, 2000, and 3000 Pa.

Each condition contains 35,000 spatial observations, giving 315,000 data rows.

**Input features:**
- Normalized spatial coordinate, x/x₀
- Yield stress, τᵧ (Pa)

**Prediction target:**
- Normalized flow depth, h/h₀

Normalization constants: x₀ = 305 and h₀ = 30.5.

Main dataset: `quadratic_yieldstress_combined_9_cases.csv`

## 3. Machine Learning Methodology

Three held-out yield-stress experiments are conducted at 50, 500, and 2000 Pa.

For each experiment, the complete target yield-stress profile is excluded from training. Both models are trained using the remaining eight profiles (280,000 observations) and evaluated against the 35,000 observations belonging to the held-out case.

Negative predicted depths are clipped to zero.

Evaluation metrics include R², RMSE, and MAE.

## 4. Installation

Requires Python 3 and the packages listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

## 5. Running the Experiments

Run the training and evaluation script:

```bash
python CV7001_Assignment_3.py
```

The script trains both models for each held-out experiment, calculates performance metrics, and generates comparison and prediction-error graphs.

## 6. Inference Demonstration

Run:

```bash
python ML_Inference_Demo.py --stress 800 --x 1000
```

Example results:

| Model | Predicted flow depth |
|---|---:|
| Polynomial Ridge | 3.756870 m |
| Random Forest | 3.732688 m |

The 800 Pa profile is excluded from training in this demonstration. The predictions illustrate inference at a specified spatial coordinate, not independently verified accuracy.

The inference script accepts yield-stress values represented in the dataset and retrains the models after excluding the selected case.

## 7. Results

| Yield stress | Model | R² | RMSE (m) |
|---|---|---:|---:|
| 50 Pa | Polynomial Ridge | 0.7657 | 0.4315 |
| 50 Pa | Random Forest | 0.9711 | 0.1514 |
| 500 Pa | Polynomial Ridge | 0.9025 | 0.4569 |
| 500 Pa | Random Forest | 0.7053 | 0.7942 |
| 2000 Pa | Polynomial Ridge | 0.9219 | 0.7571 |
| 2000 Pa | Random Forest | 0.7455 | 1.3665 |

The repository includes six figures comparing numerical simulation profiles, ML predictions, and signed prediction errors.

## 8. Limitations

Although the dataset contains 315,000 spatial observations, these represent only nine distinct yield-stress conditions.

Neither model explicitly enforces the governing flow equations or mass conservation. Prediction accuracy near sharp mudflow fronts remains an important limitation.

## 9. Reference

Tan, J. F., Tion, P. H., and Lau, S. C. (2025). *Parametric Sensitivity Analyses of Mudflow Rheology with Depth-Averaged Model*. Journal of Advanced Research in Fluid Mechanics and Thermal Sciences, 131(2), 78–94.

## 10. AI Assistance Disclosure

Generative AI assistance was used during Python code development, debugging, documentation, and report drafting. The implementation and results were reviewed by the author.

## 11. Conclusion and Reflection
The main contribution of this project is the application and evaluation of machine learning surrogate models using CFD mudflow simulation data from my previous research. The results demonstrate the potential of ML to approximate flow-depth profiles under unseen yield-stress conditions, which could reduce the computational effort required for parametric studies. In civil engineering, such models could support rapid sensitivity analysis and preliminary mudflow hazard assessment. However, the difficulty in accurately predicting sharp flow fronts highlights an important limitation, as errors in flow extent may affect inundation assessment. Future work should therefore focus on improving physical consistency and flow-front prediction accuracy before these models can be applied reliably in engineering practice.
