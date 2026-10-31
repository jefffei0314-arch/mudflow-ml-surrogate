
"""
ML Inference Demonstration
Mudflow Depth Prediction Using Polynomial Ridge and Random Forest

Example:
    python ML_Inference_Demo.py --stress 500 --x 610

The selected yield stress is excluded from training.
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor


# ============================================================
# 1. SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "quadratic_yieldstress_combined_9_cases.csv"

X0 = 305.0
H0 = 30.5

FEATURES = ["x_ratio", "yield_stress"]


# ============================================================
# 2. USER INPUTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Predict mudflow depth at an unseen yield stress."
)

parser.add_argument(
    "--stress",
    type=float,
    default=500.0,
    help="Yield stress in Pa (default: 500)"
)

parser.add_argument(
    "--x",
    type=float,
    default=1000.0,
    help="Spatial coordinate in metres (default: 1000)"
)

args = parser.parse_args()

target_stress = args.stress
x_new = args.x

if not np.isfinite(target_stress) or not np.isfinite(x_new):
    raise ValueError("Inputs must be finite numbers.")

if target_stress <= 0:
    raise ValueError("Yield stress must be positive.")


# ============================================================
# 3. LOAD CFD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

required_columns = ["yield_stress", "x", "flow_depth"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing column: {column}")

df["x_ratio"] = df["x"] / X0
df["h_ratio"] = df["flow_depth"] / H0

# Require a known held-out yield-stress case.
available_stresses = df["yield_stress"].unique()

if not np.isclose(
    available_stresses,
    target_stress
).any():
    raise ValueError(
        "For this held-out demonstration, choose a "
        "yield stress present in the combined dataset."
    )


# ============================================================
# 4. EXCLUDE THE TEST YIELD STRESS
# ============================================================

test_mask = np.isclose(
    df["yield_stress"].to_numpy(),
    target_stress
)

train = df.loc[~test_mask].copy()

if train.empty:
    raise ValueError("No training data available.")

X_train = train[FEATURES]
y_train = train["h_ratio"]

print("=" * 60)
print("MUDFLOW DEPTH INFERENCE DEMONSTRATION")
print("=" * 60)

print(f"Excluded yield stress: {target_stress:g} Pa")
print(f"Training points: {len(train):,}")

print(
    "Training yield stresses:",
    sorted(train["yield_stress"].unique().tolist())
)


# ============================================================
# 5. CREATE AND TRAIN ML MODELS
# ============================================================

poly_model = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(
        degree=4,
        include_bias=False
    ),
    Ridge(alpha=1.0)
)

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Polynomial Ridge...")
poly_model.fit(X_train, y_train)

print("Training Random Forest...")
rf_model.fit(X_train, y_train)


# ============================================================
# 6. PREPARE NEW INPUT
# ============================================================

x_ratio_new = x_new / X0

X_new = pd.DataFrame({
    "x_ratio": [x_ratio_new],
    "yield_stress": [target_stress]
})


# ============================================================
# 7. PERFORM INFERENCE
# ============================================================

poly_prediction = poly_model.predict(X_new)[0]
rf_prediction = rf_model.predict(X_new)[0]

# Prevent negative flow depths.
poly_prediction = max(poly_prediction, 0.0)
rf_prediction = max(rf_prediction, 0.0)

# Convert normalized depth to metres.
poly_depth_m = poly_prediction * H0
rf_depth_m = rf_prediction * H0


# ============================================================
# 8. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("INFERENCE INPUT")
print("=" * 60)

print(f"Yield stress       = {target_stress:g} Pa")
print(f"Spatial coordinate = {x_new:.3f} m")
print(f"Normalized x/x0    = {x_ratio_new:.5f}")

print("\n" + "=" * 60)
print("ML PREDICTIONS")
print("=" * 60)

print("\nPolynomial Ridge:")
print(f"  Normalized depth = {poly_prediction:.6f}")
print(f"  Flow depth       = {poly_depth_m:.6f} m")

print("\nRandom Forest:")
print(f"  Normalized depth = {rf_prediction:.6f}")
print(f"  Flow depth       = {rf_depth_m:.6f} m")

print("\nInference completed successfully.")

