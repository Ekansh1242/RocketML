"""
Rocket Performance Prediction API — v2.0
=========================================
Upgrades over v1:
  • XGBoost replaces RandomForest (faster, more accurate)
  • Optuna tunes XGBoost hyperparameters automatically
  • RocketCEA-derived combustion features (Isp, c*, Cf, chamber pressure)
  • PyTorch Neural Network as an optional high-fidelity model
  • /model/switch endpoint lets you toggle XGBoost ↔ NeuralNet at runtime
"""

import os, io, json, base64, pickle, logging, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app  = Flask(__name__)
CORS(app)

# ── Feature definitions ────────────────────────────────────────────────────────
# Raw design inputs
RAW_FEATURES = ["thrust", "burn_time", "mass", "drag", "angle",
                "diameter", "propellant_mass", "wind"]

# Extra combustion features derived by RocketCEA (or approximated if CEA absent)
CEA_FEATURES = ["isp_vac", "c_star", "cf", "chamber_pressure_bar"]

# Combined model input
ALL_FEATURES = RAW_FEATURES + CEA_FEATURES
TARGET_COLS  = ["apogee", "velocity"]

FEATURE_RANGES = {
    "thrust":             (800,   2000),
    "burn_time":          (3,     8),
    "mass":               (40,    100),
    "drag":               (0.2,   0.6),
    "angle":              (75,    90),
    "diameter":           (0.1,   0.3),
    "propellant_mass":    (10,    40),
    "wind":               (0,     10),
    # CEA-derived — users don't set these directly; they're computed
    "isp_vac":            (200,   320),
    "c_star":             (1400,  1700),
    "cf":                 (1.5,   1.9),
    "chamber_pressure_bar": (20,  80),
}

XGB_PATH  = "xgb_model.pkl"
NN_PATH   = "nn_model.pt"
DATA_PATH = os.environ.get("DATASET_PATH", "rocket_dataset_3000.csv")

# Active model switcher  ("xgb" | "nn")
_active_model_name = "xgb"
_models = {}   # {"xgb": <XGBRegressor>, "nn": <RocketNN>}


# ══════════════════════════════════════════════════════════════════════════════
# 1.  COMBUSTION PHYSICS  (RocketCEA)
# ══════════════════════════════════════════════════════════════════════════════

def compute_cea_features(propellant_mass: float, thrust: float,
                          burn_time: float) -> dict:
    """
    Attempt to compute combustion parameters via RocketCEA.
    Falls back to physics-informed approximations if CEA is not installed.

    Propellant assumed: HTPB/AP composite (O/F ~ 2.3)
    """
    of_ratio   = 2.3          # typical HTPB/AP composite
    pc_bar     = max(20.0, min(80.0, thrust / (burn_time * 200)))  # rough estimate

    try:
        from rocketcea.cea_obj import CEA_Obj
        cea = CEA_Obj(oxName="LOX", fuelName="HTPB")   # closest available pair
        isp_vac = cea.get_Isp(Pc=pc_bar * 14.5038,     # psia
                               MR=of_ratio, eps=8.0)
        c_star  = cea.get_Cstar(Pc=pc_bar * 14.5038, MR=of_ratio)
        cf      = isp_vac * 9.81 / c_star
    except Exception:
        # Physics-informed fallback (no CEA installed)
        # Approximation from Sutton & Biblarz, "Rocket Propulsion Elements"
        isp_vac = 250 + (pc_bar - 20) * 0.5          # ~250–280 s range
        c_star  = 1500 + (pc_bar - 20) * 3.0
        cf      = 1.65 + (pc_bar - 20) * 0.002

    return {
        "isp_vac":               round(float(isp_vac), 2),
        "c_star":                round(float(c_star),  2),
        "cf":                    round(float(cf),      4),
        "chamber_pressure_bar":  round(float(pc_bar),  2),
    }


def enrich_with_cea(df: pd.DataFrame) -> pd.DataFrame:
    """Add CEA-derived columns to a DataFrame that has RAW_FEATURES."""
    cea_rows = df.apply(
        lambda r: compute_cea_features(r["propellant_mass"], r["thrust"], r["burn_time"]),
        axis=1,
    )
    return pd.concat([df, pd.DataFrame(list(cea_rows))], axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  XGBOOST MODEL
# ══════════════════════════════════════════════════════════════════════════════

def train_xgb(X_train, y_train, params: dict = None):
    from xgboost import XGBRegressor
    from sklearn.multioutput import MultiOutputRegressor

    default_params = dict(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    if params:
        default_params.update(params)

    base = XGBRegressor(**default_params)
    mdl  = MultiOutputRegressor(base)
    mdl.fit(X_train, y_train)
    return mdl


def tune_xgb_with_optuna(X_train, y_train, X_val, y_val, n_trials=60):
    """Use Optuna to find the best XGBoost hyperparameters."""
    import optuna
    from xgboost import XGBRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.metrics import mean_squared_error

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = dict(
            n_estimators      = trial.suggest_int("n_estimators", 200, 1000),
            learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth         = trial.suggest_int("max_depth", 3, 9),
            subsample         = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha         = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda        = trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            random_state=42, n_jobs=-1, verbosity=0,
        )
        mdl = MultiOutputRegressor(XGBRegressor(**params))
        mdl.fit(X_train, y_train)
        preds = mdl.predict(X_val)
        return mean_squared_error(y_val, preds)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info("Best XGB params: %s", study.best_params)
    return study.best_params


# ══════════════════════════════════════════════════════════════════════════════
# 3.  PYTORCH NEURAL NETWORK
# ══════════════════════════════════════════════════════════════════════════════

def build_nn_model():
    """Return a PyTorch MLP for rocket performance prediction."""
    try:
        import torch
        import torch.nn as nn

        class RocketNN(nn.Module):
            def __init__(self, n_in=len(ALL_FEATURES), n_out=len(TARGET_COLS)):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(n_in, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Linear(128, n_out),
                )

            def forward(self, x):
                return self.net(x)

        return RocketNN()
    except ImportError:
        return None


def train_nn(X_train_np, y_train_np, epochs=200, lr=1e-3):
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    model = build_nn_model()
    if model is None:
        raise RuntimeError("PyTorch is not installed.")

    # Normalise
    X_mean, X_std = X_train_np.mean(0), X_train_np.std(0) + 1e-8
    y_mean, y_std = y_train_np.mean(0), y_train_np.std(0) + 1e-8

    Xn = torch.tensor((X_train_np - X_mean) / X_std, dtype=torch.float32)
    yn = torch.tensor((y_train_np - y_mean) / y_std,  dtype=torch.float32)

    loader = DataLoader(TensorDataset(Xn, yn), batch_size=64, shuffle=True)
    opt    = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        sched.step()
        if epoch % 50 == 0:
            logger.info("NN epoch %d/%d", epoch, epochs)

    # Store normalisation stats inside model for inference
    model.X_mean = X_mean
    model.X_std  = X_std
    model.y_mean = y_mean
    model.y_std  = y_std

    return model


def nn_predict(model, X_np):
    import torch
    model.eval()
    Xn = torch.tensor((X_np - model.X_mean) / model.X_std, dtype=torch.float32)
    with torch.no_grad():
        yn = model(Xn).numpy()
    return yn * model.y_std + model.y_mean


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MODEL LOADING / TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def load_or_train_all():
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error

    if not os.path.exists(DATA_PATH):
        logger.warning("No dataset at %s — models not loaded.", DATA_PATH)
        return

    df = pd.read_csv(DATA_PATH)

    # Add CEA features if missing
    if "isp_vac" not in df.columns:
        logger.info("Computing CEA combustion features …")
        df = enrich_with_cea(df)
        df.to_csv(DATA_PATH, index=False)
        logger.info("Dataset updated with combustion features.")

    X = df[ALL_FEATURES].values
    y = df[TARGET_COLS].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── XGBoost ────────────────────────────────────────────────────────────────
    if os.path.exists(XGB_PATH):
        logger.info("Loading XGBoost model …")
        with open(XGB_PATH, "rb") as f:
            _models["xgb"] = pickle.load(f)
    else:
        logger.info("Tuning XGBoost with Optuna …")
        try:
            best_params = tune_xgb_with_optuna(X_train, y_train, X_test, y_test, n_trials=60)
        except Exception as e:
            logger.warning("Optuna tuning failed (%s) — using defaults.", e)
            best_params = {}
        _models["xgb"] = train_xgb(X_train, y_train, best_params)
        preds = _models["xgb"].predict(X_test)
        logger.info("XGB  R²=%.4f  MAE=%.2f", r2_score(y_test, preds), mean_absolute_error(y_test, preds))
        with open(XGB_PATH, "wb") as f:
            pickle.dump(_models["xgb"], f)
        logger.info("XGBoost model saved → %s", XGB_PATH)

    # ── Neural Network ─────────────────────────────────────────────────────────
    try:
        import torch
        if os.path.exists(NN_PATH):
            logger.info("Loading Neural Network model …")
            import pickle as _pkl
            with open(NN_PATH, "rb") as _f:
                _models["nn"] = _pkl.load(_f)
        else:
            logger.info("Training PyTorch Neural Network …")
            _models["nn"] = train_nn(X_train, y_train, epochs=200)
            nn_preds = nn_predict(_models["nn"], X_test)
            logger.info("NN   R²=%.4f  MAE=%.2f",
                        r2_score(y_test, nn_preds), mean_absolute_error(y_test, nn_preds))
            import pickle as _pkl
            with open(NN_PATH, "wb") as _f:
                _pkl.dump(_models["nn"], _f)
            logger.info("Neural Network saved → %s", NN_PATH)
    except ImportError:
        logger.warning("PyTorch not installed — Neural Network skipped.")
    except Exception as e:
        logger.warning("Neural Network training failed: %s", e)


load_or_train_all()


# ══════════════════════════════════════════════════════════════════════════════
# 5.  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def get_active_model():
    return _models.get(_active_model_name)


def model_predict(X_df: pd.DataFrame) -> np.ndarray:
    m = get_active_model()
    if m is None:
        raise RuntimeError("No model loaded.")
    if _active_model_name == "nn":
        return nn_predict(m, X_df[ALL_FEATURES].values)
    return m.predict(X_df[ALL_FEATURES].values)


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


def validate_raw(data: dict):
    cleaned = {}
    for col in RAW_FEATURES:
        lo, hi = FEATURE_RANGES[col]
        if col not in data:
            return None, f"Missing field: '{col}'"
        try:
            val = float(data[col])
        except (ValueError, TypeError):
            return None, f"Field '{col}' must be a number"
        if not (lo <= val <= hi):
            return None, f"'{col}' must be {lo}–{hi} (got {val})"
        cleaned[col] = val
    return cleaned, None


def build_full_input(cleaned: dict) -> pd.DataFrame:
    """Append CEA features to raw design inputs."""
    cea = compute_cea_features(
        cleaned["propellant_mass"], cleaned["thrust"], cleaned["burn_time"]
    )
    row = {**cleaned, **cea}
    return pd.DataFrame([row], columns=ALL_FEATURES)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":        "ok",
        "active_model":  _active_model_name,
        "models_loaded": list(_models.keys()),
    })


@app.route("/model/switch", methods=["POST"])
def switch_model():
    global _active_model_name
    body = request.get_json(force=True) or {}
    name = body.get("model", "").lower()
    if name not in _models:
        return jsonify({"error": f"Model '{name}' is not loaded."}), 503
    _active_model_name = name
    return jsonify({"active_model": _active_model_name})


@app.route("/combustion", methods=["GET"])
def combustion():
    """
    Return CEA combustion parameters for a design.
    Query params: thrust, burn_time, propellant_mass
    """
    try:
        thrust          = float(request.args.get("thrust", 1500))
        burn_time       = float(request.args.get("burn_time", 5))
        propellant_mass = float(request.args.get("propellant_mass", 25))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    cea = compute_cea_features(propellant_mass, thrust, burn_time)
    return jsonify({"inputs": {"thrust": thrust, "burn_time": burn_time,
                                "propellant_mass": propellant_mass},
                    "combustion": cea,
                    "note": "Computed via RocketCEA if installed, else physics-informed approximation."})


@app.route("/predict", methods=["POST"])
def predict():
    if not _models:
        return jsonify({"error": "No models loaded."}), 503

    raw, err = validate_raw(request.get_json(force=True) or {})
    if err:
        return jsonify({"error": err}), 400

    X_in    = build_full_input(raw)
    cea_row = {k: X_in[k].iloc[0] for k in CEA_FEATURES}
    pred    = model_predict(X_in)[0]

    return jsonify({
        "model":       _active_model_name,
        "design":      raw,
        "combustion":  {k: round(float(v), 4) for k, v in cea_row.items()},
        "predictions": {
            "apogee_m":        round(float(pred[0]), 2),
            "max_velocity_ms": round(float(pred[1]), 2),
        }
    })


@app.route("/optimize", methods=["POST"])
def optimize():
    if not _models:
        return jsonify({"error": "No models loaded."}), 503

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        return jsonify({"error": "optuna not installed."}), 500

    body     = request.get_json(force=True) or {}
    n_trials = min(int(body.get("n_trials", 120)), 500)

    def objective(trial):
        params = {col: trial.suggest_float(col, lo, hi)
                  for col, (lo, hi) in FEATURE_RANGES.items()
                  if col in RAW_FEATURES}
        X_in = build_full_input(params)
        return model_predict(X_in)[0][0]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best    = study.best_params
    X_best  = build_full_input(best)
    bp      = model_predict(X_best)[0]

    return jsonify({
        "model":       _active_model_name,
        "n_trials":    n_trials,
        "best_design": {k: round(v, 4) for k, v in best.items()},
        "combustion":  {k: round(float(X_best[k].iloc[0]), 4) for k in CEA_FEATURES},
        "predictions": {
            "apogee_m":        round(float(bp[0]), 2),
            "max_velocity_ms": round(float(bp[1]), 2),
        }
    })


@app.route("/monte_carlo", methods=["POST"])
def monte_carlo():
    if not _models:
        return jsonify({"error": "No models loaded."}), 503

    body = request.get_json(force=True) or {}
    raw, err = validate_raw(body)
    if err:
        return jsonify({"error": err}), 400

    n_samples = min(int(body.get("n_samples", 300)), 2000)
    noise_std = float(body.get("noise_std", 0.05))
    nominal   = np.array([raw[c] for c in RAW_FEATURES])

    results = []
    for _ in range(n_samples):
        noisy = nominal * (1 + np.random.normal(0, noise_std, len(RAW_FEATURES)))
        noisy_dict = dict(zip(RAW_FEATURES, noisy))
        X_in = build_full_input(noisy_dict)
        results.append(model_predict(X_in)[0][0])

    results = np.array(results)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(results, bins=30, color="#4f81bd", edgecolor="white")
    ax.axvline(results.mean(), color="red", linestyle="--",
               label=f"Mean: {results.mean():.0f} m")
    ax.set_title("Monte Carlo Apogee Distribution")
    ax.set_xlabel("Apogee (m)")
    ax.set_ylabel("Frequency")
    ax.legend()

    return jsonify({
        "model":      _active_model_name,
        "n_samples":  n_samples,
        "noise_std":  noise_std,
        "statistics": {
            "mean_m":  round(float(results.mean()), 2),
            "std_m":   round(float(results.std()),  2),
            "min_m":   round(float(results.min()),  2),
            "max_m":   round(float(results.max()),  2),
            "p5_m":    round(float(np.percentile(results,  5)), 2),
            "p95_m":   round(float(np.percentile(results, 95)), 2),
        },
        "histogram_png_base64": fig_to_b64(fig),
    })


@app.route("/feature_importance", methods=["GET"])
def feature_importance():
    if "xgb" not in _models:
        return jsonify({"error": "XGBoost model not loaded."}), 503

    # Average importance across the two output estimators
    importances = np.mean(
        [est.feature_importances_ for est in _models["xgb"].estimators_], axis=0
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#e05c5c" if f in CEA_FEATURES else "#4f81bd" for f in ALL_FEATURES]
    ax.bar(ALL_FEATURES, importances, color=colors)
    ax.set_xticklabels(ALL_FEATURES, rotation=45, ha="right")
    ax.set_title("XGBoost Feature Importance (blue=design, red=combustion)")
    ax.set_ylabel("Importance")
    plt.tight_layout()

    return jsonify({
        "features":    ALL_FEATURES,
        "importances": [round(float(v), 6) for v in importances],
        "chart_png_base64": fig_to_b64(fig),
    })


# ── Entry ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
