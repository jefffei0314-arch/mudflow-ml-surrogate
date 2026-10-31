
"""
Machine Learning Prediction of Mudflow Depth Profiles

Three held-out yield-stress experiments:
    50 Pa, 500 Pa, 2000 Pa

Models:
    1. Polynomial Ridge Regression
    2. Random Forest Regression

All figures and CSV results are saved in ML_Results.
No graph windows will appear.

Required packages:
    pip install numpy pandas matplotlib scikit-learn
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save figures without opening windows

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)


# ============================================================
# 1. SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "quadratic_yieldstress_combined_9_cases.csv"

REFERENCE_FILES = {
    50: BASE_DIR / "quadratic_50_clean.csv",
    500: BASE_DIR / "quadratic_500_clean.csv",
    2000: BASE_DIR / "quadratic_2000_clean.csv",
}

OUTPUT_DIR = BASE_DIR / "ML_Results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

X0 = 305.0
H0 = 30.5

FEATURES = ["x_ratio", "yield_stress"]


# ============================================================
# 2. LOAD DATA
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Combined dataset not found: {DATA_FILE}"
    )

df = pd.read_csv(DATA_FILE)

required_columns = ["yield_stress", "x", "flow_depth"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Missing column in combined dataset: {column}"
        )

df["x_ratio"] = df["x"] / X0
df["h_ratio"] = df["flow_depth"] / H0

print("=" * 65)
print("MACHINE LEARNING MUDFLOW DEPTH PREDICTION")
print("=" * 65)

print(
    "Available yield stresses:",
    sorted(df["yield_stress"].unique().tolist())
)

print(f"Total data points: {len(df):,}")


# ============================================================
# 3. DEFINE MODELS
# ============================================================

def create_models():

    polynomial_ridge = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(
            degree=4,
            include_bias=False
        ),
        Ridge(alpha=1.0),
    )

    random_forest = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )

    return polynomial_ridge, random_forest


# ============================================================
# 4. PERFORMANCE METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE_h_ratio": np.sqrt(mse),
        "MAE_h_ratio": mae,
        "RMSE_m": np.sqrt(mse) * H0,
        "MAE_m": mae * H0,
    }


# ============================================================
# 5. FLOW PROFILE COMPARISON GRAPH
# ============================================================

def plot_flow_profile(
    yield_stress,
    x,
    y_true,
    y_poly,
    y_rf,
):

    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.plot(
        x,
        y_true,
        color="tab:blue",
        linewidth=2.2,
        label="Numerical Simulation",
    )

    ax.plot(
        x,
        y_poly,
        color="tab:orange",
        linestyle="--",
        linewidth=1.8,
        label="Polynomial Ridge",
    )

    ax.plot(
        x,
        y_rf,
        color="tab:green",
        linestyle="-.",
        linewidth=1.8,
        label="Random Forest",
    )

    ax.set_title(
        f"Flow Depth Profile Comparison - "
        f"Yield Stress = {yield_stress} Pa"
    )

    ax.set_xlabel("Normalized Distance (x/x0)")
    ax.set_ylabel("Normalized Flow Depth (h/h0)")

    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    filename = (
        OUTPUT_DIR
        / f"flow_profile_yieldstress_{yield_stress}.png"
    )

    fig.savefig(filename, dpi=300)
    plt.close(fig)

    print(f"Saved: {filename.name}")


# ============================================================
# 6. PREDICTION ERROR GRAPH
# ============================================================

def plot_prediction_error(
    yield_stress,
    x,
    y_true,
    y_poly,
    y_rf,
):

    error_poly = y_poly - y_true
    error_rf = y_rf - y_true

    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.axhline(
        0,
        color="black",
        linestyle=":",
        linewidth=1.2,
    )

    ax.plot(
        x,
        error_poly,
        color="tab:orange",
        linewidth=1.8,
        label="Polynomial Ridge",
    )

    ax.plot(
        x,
        error_rf,
        color="tab:green",
        linewidth=1.8,
        label="Random Forest",
    )

    ax.set_title(
        f"Prediction Error - "
        f"Yield Stress = {yield_stress} Pa"
    )

    ax.set_xlabel("Normalized Distance (x/x0)")

    ax.set_ylabel(
        "Signed Error in Normalized Flow Depth"
    )

    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    filename = (
        OUTPUT_DIR
        / f"prediction_error_yieldstress_{yield_stress}.png"
    )

    fig.savefig(filename, dpi=300)
    plt.close(fig)

    print(f"Saved: {filename.name}")


# ============================================================
# 7. RUN THREE EXPERIMENTS
# ============================================================

summary = []

for target_stress, reference_file in REFERENCE_FILES.items():

    print("\n" + "=" * 65)
    print(f"HELD-OUT YIELD STRESS: {target_stress} Pa")
    print("=" * 65)

    if not reference_file.exists():
        raise FileNotFoundError(
            f"Reference file not found: {reference_file}"
        )

    # --------------------------------------------------------
    # 7.1 SEPARATE TRAINING AND TESTING DATA
    # --------------------------------------------------------

    test_mask = np.isclose(
        df["yield_stress"].to_numpy(),
        target_stress
    )

    train = df.loc[~test_mask].copy()
    held_out = df.loc[test_mask].copy()

    if held_out.empty:
        raise ValueError(
            f"No held-out data for {target_stress} Pa"
        )

    if np.isclose(
        train["yield_stress"].to_numpy(),
        target_stress
    ).any():
        raise ValueError("Data leakage detected!")

    # --------------------------------------------------------
    # 7.2 LOAD REFERENCE NUMERICAL SIMULATION
    # --------------------------------------------------------

    reference = pd.read_csv(reference_file)

    for column in ["x", "flow_depth"]:
        if column not in reference.columns:
            raise ValueError(
                f"{reference_file.name} missing {column}"
            )

    if "yield_stress" in reference.columns:
        if not np.isclose(
            reference["yield_stress"].to_numpy(),
            target_stress
        ).all():
            raise ValueError(
                f"Wrong yield stress in {reference_file.name}"
            )

    reference["x_ratio"] = reference["x"] / X0
    reference["h_ratio"] = reference["flow_depth"] / H0

    reference = (
        reference.sort_values("x_ratio")
        .reset_index(drop=True)
    )

    # Verify reference against the held-out combined data.

    check = (
        held_out.sort_values("x")
        .reset_index(drop=True)
    )

    if len(check) != len(reference):
        raise ValueError(
            f"Point count mismatch at {target_stress} Pa"
        )

    if not np.allclose(
        check["x"].to_numpy(),
        reference["x"].to_numpy(),
        atol=1e-8,
        rtol=0,
    ):
        raise ValueError(
            f"Spatial grid mismatch at {target_stress} Pa"
        )

    max_depth_difference = np.max(
        np.abs(
            check["flow_depth"].to_numpy()
            - reference["flow_depth"].to_numpy()
        )
    )

    if not np.allclose(
        check["flow_depth"].to_numpy(),
        reference["flow_depth"].to_numpy(),
        atol=1e-8,
        rtol=0,
    ):
        raise ValueError(
            f"Depth mismatch at {target_stress} Pa: "
            f"{max_depth_difference}"
        )

    print(
        f"Reference verified. "
        f"Maximum depth difference: "
        f"{max_depth_difference:.3g} m"
    )

    # --------------------------------------------------------
    # 7.3 PREPARE INPUTS
    # --------------------------------------------------------

    X_train = train[FEATURES]
    y_train = train["h_ratio"]

    X_test = pd.DataFrame({
        "x_ratio": reference["x_ratio"].to_numpy(),
        "yield_stress": np.full(
            len(reference),
            float(target_stress)
        ),
    })

    y_true = reference["h_ratio"].to_numpy()
    x_plot = reference["x_ratio"].to_numpy()

    print(
        "Training stresses:",
        sorted(train["yield_stress"].unique().tolist())
    )

    print(f"Training points: {len(train):,}")
    print(f"Testing points: {len(reference):,}")

    # --------------------------------------------------------
    # 7.4 TRAIN MODELS
    # --------------------------------------------------------

    poly_model, rf_model = create_models()

    print("Training Polynomial Ridge...")
    poly_model.fit(X_train, y_train)

    print("Training Random Forest...")
    rf_model.fit(X_train, y_train)

    # --------------------------------------------------------
    # 7.5 PREDICT
    # --------------------------------------------------------

    y_poly = poly_model.predict(X_test)
    y_rf = rf_model.predict(X_test)

    # Flow depth cannot be negative.
    y_poly = np.clip(y_poly, 0, None)
    y_rf = np.clip(y_rf, 0, None)

    # --------------------------------------------------------
    # 7.6 EVALUATE
    # --------------------------------------------------------

    results = {
        "Polynomial Ridge": y_poly,
        "Random Forest": y_rf,
    }

    for model_name, prediction in results.items():

        metrics = calculate_metrics(
            y_true,
            prediction
        )

        summary.append({
            "yield_stress_Pa": target_stress,
            "model": model_name,
            "training_points": len(train),
            "testing_points": len(reference),
            **metrics,
        })

        print(f"\n{model_name}")
        print(f"R2       = {metrics['R2']:.5f}")
        print(f"RMSE (m) = {metrics['RMSE_m']:.5f}")
        print(f"MAE (m)  = {metrics['MAE_m']:.5f}")

    # --------------------------------------------------------
    # 7.7 SAVE COMPARISON CSV
    # --------------------------------------------------------

    comparison = pd.DataFrame({
        "yield_stress_Pa": target_stress,
        "x_m": reference["x"].to_numpy(),
        "x_ratio": x_plot,
        "Numerical_Simulation_h_ratio": y_true,
        "Polynomial_Ridge_h_ratio": y_poly,
        "Random_Forest_h_ratio": y_rf,
        "Numerical_Simulation_depth_m": y_true * H0,
        "Polynomial_Ridge_depth_m": y_poly * H0,
        "Random_Forest_depth_m": y_rf * H0,
        "Polynomial_Ridge_error_h_ratio": (
            y_poly - y_true
        ),
        "Random_Forest_error_h_ratio": (
            y_rf - y_true
        ),
    })

    comparison_file = (
        OUTPUT_DIR
        / f"comparison_yieldstress_{target_stress}.csv"
    )

    comparison.to_csv(
        comparison_file,
        index=False
    )

    print(f"\nSaved: {comparison_file.name}")

    # --------------------------------------------------------
    # 7.8 SAVE GRAPHS WITHOUT DISPLAYING THEM
    # --------------------------------------------------------

    plot_flow_profile(
        target_stress,
        x_plot,
        y_true,
        y_poly,
        y_rf,
    )

    plot_prediction_error(
        target_stress,
        x_plot,
        y_true,
        y_poly,
        y_rf,
    )


# ============================================================
# 8. SAVE OVERALL PERFORMANCE SUMMARY
# ============================================================

summary_df = pd.DataFrame(summary)

summary_file = (
    OUTPUT_DIR / "model_performance_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)

print("\n" + "=" * 65)
print("OVERALL MODEL PERFORMANCE")
print("=" * 65)

print(
    summary_df[
        [
            "yield_stress_Pa",
            "model",
            "R2",
            "RMSE_m",
            "MAE_m",
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 65)
print("ALL EXPERIMENTS COMPLETED")
print("=" * 65)

print(f"Results saved to: {OUTPUT_DIR}")
print("Six PNG graphs generated.")
print("Four CSV files generated.")
print("No graph windows were opened.")
