
"""
Inference using the validated CV7001 mudflow models.

Features:
    1. Train Polynomial Ridge and Random Forest using validated parameters.
    2. Exclude the selected yield-stress profile from training.
    3. Predict mudflow depth at a user-specified spatial coordinate.
    4. Retrieve the numerical reference from the held-out profile.
    5. Interpolate the numerical reference when necessary.
    6. Calculate absolute and percentage prediction errors.

Required files:
    quadratic_yieldstress_combined_9_cases.csv
    ML_Results/Validation/validation_summary.csv

Example:
    python ML_Inference_Demo_Updated.py --stress 500 --x 610
"""

from pathlib import Path
import argparse
import ast

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor


# ============================================================
# 1. SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "quadratic_yieldstress_combined_9_cases.csv"

VALIDATION_FILE = (
    BASE_DIR / "ML_Results" / "Validation" / "validation_summary.csv"
)

X0 = 305.0
H0 = 30.5

FEATURES = ["x_ratio", "yield_stress"]

TEST_STRESSES = (50, 500, 2000)


# ============================================================
# 2. READ VALIDATED HYPERPARAMETERS
# ============================================================

def validated_parameters(summary, stress, model_name):

    rows = summary.loc[
        np.isclose(
            summary["yield_stress_Pa"].astype(float),
            stress
        )
        & summary["model"].eq(model_name)
    ]

    if len(rows) != 1:
        raise ValueError(
            f"Expected one validation row for "
            f"{stress:g} Pa / {model_name}; found {len(rows)}"
        )

    params = ast.literal_eval(
        rows.iloc[0]["best_parameters"]
    )

    if not isinstance(params, dict):
        raise ValueError(
            "Validation parameters must be a dictionary."
        )

    return params


# ============================================================
# 3. NUMERICAL REFERENCE AT REQUESTED LOCATION
# ============================================================

def numerical_reference(held_out, x):

    reference = (
        held_out[["x", "flow_depth"]]
        .sort_values("x")
    )

    x_values = reference["x"].to_numpy(dtype=float)
    h_values = reference["flow_depth"].to_numpy(dtype=float)

    if len(x_values) == 0:
        return None

    # Ensure that interpolation uses unique coordinates.
    if np.any(np.diff(x_values) == 0):
        raise ValueError(
            "Duplicate spatial coordinates exist in "
            "the numerical reference profile."
        )

    x_min = x_values.min()
    x_max = x_values.max()

    # Do not extrapolate outside the numerical dataset.
    if x < x_min or x > x_max:
        return None

    # Linear interpolation between neighbouring data points.
    h_numerical = np.interp(
        x,
        x_values,
        h_values
    )

    return float(h_numerical)


# ============================================================
# 4. ERROR CALCULATION
# ============================================================

def calculate_errors(predicted, reference):

    absolute_error = abs(predicted - reference)

    if not np.isclose(reference, 0.0, atol=1e-12):
        percentage_error = (
            absolute_error / abs(reference)
        ) * 100.0
    else:
        percentage_error = None

    return absolute_error, percentage_error


# ============================================================
# 5. MAIN INFERENCE PROGRAM
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Infer mudflow depth using validated "
            "held-out ML models."
        )
    )

    parser.add_argument(
        "--stress",
        type=float,
        default=500.0,
        help="Held-out yield stress: 50, 500 or 2000 Pa"
    )

    parser.add_argument(
        "--x",
        type=float,
        default=610.0,
        help="Spatial coordinate (m)"
    )

    args = parser.parse_args()

    stress = args.stress
    x = args.x

    # --------------------------------------------------------
    # 5.1 Input validation
    # --------------------------------------------------------

    if not np.isfinite(stress) or not np.isfinite(x):
        parser.error("Inputs must be finite.")

    if stress not in TEST_STRESSES:
        parser.error(
            "Choose a validated held-out stress: "
            "50, 500 or 2000 Pa."
        )

    if x < 0:
        parser.error(
            "Spatial coordinate must be nonnegative."
        )

    for path in (DATA_FILE, VALIDATION_FILE):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing required file: {path}"
            )

    # --------------------------------------------------------
    # 5.2 Load numerical simulation dataset
    # --------------------------------------------------------

    df = pd.read_csv(DATA_FILE)

    required = {"yield_stress", "x", "flow_depth"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"Missing dataset columns: "
            f"{sorted(required - set(df.columns))}"
        )

    if not np.isfinite(
        df[["yield_stress", "x", "flow_depth"]].to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "Dataset contains missing or non-finite values."
        )

    df["x_ratio"] = df["x"] / X0
    df["h_ratio"] = df["flow_depth"] / H0

    # --------------------------------------------------------
    # 5.3 Separate training and held-out reference data
    # --------------------------------------------------------

    held_out_mask = np.isclose(
        df["yield_stress"].to_numpy(),
        stress
    )

    if not held_out_mask.any():
        raise ValueError(
            f"No {stress:g} Pa profile exists in the dataset."
        )

    train = df.loc[~held_out_mask].copy()
    held_out = df.loc[held_out_mask].copy()

    if train.empty:
        raise ValueError("Training data are empty.")

    X_train = train[FEATURES]
    y_train = train["h_ratio"]

    X_new = pd.DataFrame({
        "x_ratio": [x / X0],
        "yield_stress": [stress]
    })

    # --------------------------------------------------------
    # 5.4 Retrieve numerical reference
    # --------------------------------------------------------

    h_numerical = numerical_reference(
        held_out,
        x
    )

    # --------------------------------------------------------
    # 5.5 Load validated hyperparameters
    # --------------------------------------------------------

    summary = pd.read_csv(VALIDATION_FILE)

    for col in (
        "yield_stress_Pa",
        "model",
        "best_parameters"
    ):
        if col not in summary.columns:
            raise ValueError(
                f"Missing validation summary column: {col}"
            )

    ridge_params = validated_parameters(
        summary,
        stress,
        "Polynomial Ridge"
    )

    forest_params = validated_parameters(
        summary,
        stress,
        "Random Forest"
    )

    # --------------------------------------------------------
    # 5.6 Construct models
    # --------------------------------------------------------

    ridge = Pipeline([
        ("scale", StandardScaler()),
        (
            "poly",
            PolynomialFeatures(include_bias=False)
        ),
        ("poly_scale", StandardScaler()),
        ("ridge", Ridge()),
    ])

    ridge.set_params(**ridge_params)

    forest = Pipeline([
        (
            "rf",
            RandomForestRegressor(
                random_state=42,
                n_jobs=1
            )
        ),
    ])

    forest.set_params(**forest_params)

    # --------------------------------------------------------
    # 5.7 Display inference information
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("VALIDATED MUDFLOW DEPTH INFERENCE DEMONSTRATION")
    print("=" * 65)

    print(f"Excluded test stress : {stress:g} Pa")
    print(f"Training points      : {len(train):,}")
    print(
        f"Training stresses    : "
        f"{sorted(train['yield_stress'].unique().tolist())}"
    )

    print(f"Spatial coordinate   : {x:.3f} m")
    print(f"Normalized position  : {x / X0:.6f}")

    # --------------------------------------------------------
    # 5.8 Train models and perform inference
    # --------------------------------------------------------

    predictions = {}

    for name, model, params in (
        ("Polynomial Ridge", ridge, ridge_params),
        ("Random Forest", forest, forest_params),
    ):

        print(f"\nTraining {name}")
        print(f"Validated parameters: {params}")

        model.fit(X_train, y_train)

        h_ratio = max(
            float(model.predict(X_new)[0]),
            0.0
        )

        h_predicted = h_ratio * H0

        predictions[name] = {
            "h_ratio": h_ratio,
            "depth": h_predicted
        }

    # --------------------------------------------------------
    # 5.9 Display numerical reference
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("NUMERICAL SIMULATION REFERENCE")
    print("=" * 65)

    if h_numerical is not None:

        print(
            f"Numerical depth : {h_numerical:.6f} m"
        )

        print(
            f"Normalized depth: "
            f"{h_numerical / H0:.6f}"
        )

    else:

        print(
            "Numerical reference is unavailable at "
            "the requested coordinate."
        )

        print(
            "The coordinate is outside the numerical "
            "reference profile range."
        )

    # --------------------------------------------------------
    # 5.10 Display predictions and errors
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("ML PREDICTIONS AND COMPARISON")
    print("=" * 65)

    for name, result in predictions.items():

        print(f"\n{name}")

        print(
            f"Predicted normalized depth : "
            f"{result['h_ratio']:.6f}"
        )

        print(
            f"Predicted depth (m)        : "
            f"{result['depth']:.6f}"
        )

        if h_numerical is not None:

            absolute_error, percentage_error = (
                calculate_errors(
                    result["depth"],
                    h_numerical
                )
            )

            print(
                f"Absolute error (m)         : "
                f"{absolute_error:.6f}"
            )

            if percentage_error is not None:

                print(
                    f"Percentage error (%)       : "
                    f"{percentage_error:.2f}"
                )

            else:

                print(
                    "Percentage error (%)       : "
                    "N/A (zero numerical depth)"
                )

        else:

            print(
                "Prediction error            : "
                "N/A (no numerical reference)"
            )

    print("\n" + "=" * 65)
    print("INFERENCE COMPLETED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    main()
