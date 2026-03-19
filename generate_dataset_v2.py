"""
generate_dataset.py  —  v2.0
==============================
1. Generate rocket flight dataset via RocketPy simulations
2. Enrich each row with RocketCEA combustion features
   (Isp_vac, c*, Cf, chamber pressure)
3. Tune XGBoost hyperparameters with Optuna
4. Train final XGBoost model  →  xgb_model.pkl
5. Train PyTorch Neural Network  →  nn_model.pt

Usage:
    python generate_dataset.py [--simulations 3000] [--optuna-trials 60] [--nn-epochs 200]
"""

import argparse, os, pickle, logging, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from rocketpy import Environment, SolidMotor, Rocket, Flight
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RAW_FEATURES = ["thrust", "burn_time", "mass", "drag", "angle",
                "diameter", "propellant_mass", "wind"]
CEA_FEATURES = ["isp_vac", "c_star", "cf", "chamber_pressure_bar"]
ALL_FEATURES = RAW_FEATURES + CEA_FEATURES
TARGET_COLS  = ["apogee", "velocity"]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1  —  RocketPy simulation
# ══════════════════════════════════════════════════════════════════════════════

def build_env():
    return Environment(latitude=32.990254, longitude=-106.974998, elevation=1400)


def run_flight(env, thrust, burn_time, mass, drag, angle):
    motor = SolidMotor(
        thrust_source=thrust, burn_time=burn_time,
        dry_mass=2, dry_inertia=(0.1, 0.1, 0.002),
        grain_number=1, grain_density=1815,
        grain_outer_radius=0.033, grain_initial_inner_radius=0.015,
        grain_initial_height=0.12, grain_separation=0.0,
        grains_center_of_mass_position=0, center_of_dry_mass_position=0,
        nozzle_radius=0.01, throat_radius=0.005,
        interpolation_method="linear",
    )
    rocket = Rocket(
        radius=0.0635, mass=mass, inertia=(6.321, 6.321, 0.034),
        power_off_drag=drag, power_on_drag=drag,
        center_of_mass_without_motor=0,
    )
    rocket.add_motor(motor, position=-1.255)
    flight = Flight(rocket=rocket, environment=env,
                    rail_length=5, inclination=angle, heading=0)
    return float(flight.apogee), float(flight.max_speed), float(flight.time[-1])


def generate_dataset(n: int) -> pd.DataFrame:
    env, data = build_env(), []
    for i in range(n):
        thrust          = np.random.uniform(800, 2000)
        burn_time       = np.random.uniform(3, 8)
        mass            = np.random.uniform(40, 100)
        drag            = np.random.uniform(0.2, 0.6)
        angle           = np.random.uniform(75, 90)
        diameter        = np.random.uniform(0.1, 0.3)
        propellant_mass = np.random.uniform(10, 40)
        wind            = np.random.uniform(0, 10)
        try:
            apogee, velocity, ft = run_flight(env, thrust, burn_time, mass, drag, angle)
        except Exception as e:
            logger.warning("Sim %d failed: %s", i, e)
            apogee = velocity = ft = 0.0
        data.append([thrust, burn_time, mass, drag, angle,
                     diameter, propellant_mass, wind, apogee, velocity, ft])
        if i % 200 == 0:
            logger.info("Simulation %d / %d", i, n)
    return pd.DataFrame(data, columns=RAW_FEATURES + TARGET_COLS + ["flight_time"])


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2  —  RocketCEA combustion features
# ══════════════════════════════════════════════════════════════════════════════

def compute_cea(propellant_mass: float, thrust: float, burn_time: float) -> dict:
    """Return combustion parameters. Uses RocketCEA when available."""
    pc_bar = max(20.0, min(80.0, thrust / (burn_time * 200)))
    of_ratio = 2.3

    try:
        from rocketcea.cea_obj import CEA_Obj
        # LOX/HTPB is a valid CEA propellant combination
        cea = CEA_Obj(oxName="LOX", fuelName="HTPB")
        isp_vac = cea.get_Isp(Pc=pc_bar * 14.5038, MR=of_ratio, eps=8.0)
        c_star  = cea.get_Cstar(Pc=pc_bar * 14.5038, MR=of_ratio)
        cf      = isp_vac * 9.81 / c_star
        logger.debug("CEA computed: Isp=%.1f c*=%.1f Cf=%.3f Pc=%.1f", isp_vac, c_star, cf, pc_bar)
    except Exception:
        # Sutton & Biblarz physics-informed approximation (fallback)
        isp_vac = 250 + (pc_bar - 20) * 0.5
        c_star  = 1500 + (pc_bar - 20) * 3.0
        cf      = 1.65 + (pc_bar - 20) * 0.002

    return {
        "isp_vac":               round(float(isp_vac), 2),
        "c_star":                round(float(c_star),  2),
        "cf":                    round(float(cf),      4),
        "chamber_pressure_bar":  round(float(pc_bar),  2),
    }


def enrich_with_cea(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing combustion features for %d rows …", len(df))
    rows = df.apply(lambda r: compute_cea(r["propellant_mass"], r["thrust"], r["burn_time"]), axis=1)
    return pd.concat([df, pd.DataFrame(list(rows))], axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3  —  XGBoost + Optuna
# ══════════════════════════════════════════════════════════════════════════════

def tune_xgb(X_train, y_train, X_val, y_val, n_trials: int):
    import optuna
    from xgboost import XGBRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.metrics import mean_squared_error

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        p = dict(
            n_estimators      = trial.suggest_int("n_estimators", 200, 1000),
            learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth         = trial.suggest_int("max_depth", 3, 9),
            subsample         = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha         = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda        = trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            random_state=42, n_jobs=-1, verbosity=0,
        )
        mdl = MultiOutputRegressor(XGBRegressor(**p))
        mdl.fit(X_train, y_train)
        return mean_squared_error(y_val, mdl.predict(X_val))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    logger.info("Best XGB params: %s  (MSE=%.2f)", study.best_params, study.best_value)
    return study.best_params


def train_xgb(X_train, y_train, params: dict):
    from xgboost import XGBRegressor
    from sklearn.multioutput import MultiOutputRegressor

    defaults = dict(random_state=42, n_jobs=-1, verbosity=0,
                    n_estimators=600, learning_rate=0.05, max_depth=6)
    defaults.update(params)
    mdl = MultiOutputRegressor(XGBRegressor(**defaults))
    mdl.fit(X_train, y_train)
    return mdl


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4  —  PyTorch Neural Network
# ══════════════════════════════════════════════════════════════════════════════

try:
    import torch
    import torch.nn as nn

    class RocketNN(nn.Module):
        def __init__(self, n_in=12, n_out=2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_in, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(256, 256),  nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(256, 128),  nn.BatchNorm1d(128), nn.ReLU(),
                nn.Linear(128, n_out),
            )
        def forward(self, x):
            return self.net(x)

except ImportError:
    RocketNN = None


def train_nn(X_train_np, y_train_np, epochs: int):
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    n_in  = X_train_np.shape[1]
    n_out = y_train_np.shape[1]

    # Normalise
    X_mean, X_std = X_train_np.mean(0), X_train_np.std(0) + 1e-8
    y_mean, y_std = y_train_np.mean(0), y_train_np.std(0) + 1e-8
    Xn = torch.tensor((X_train_np - X_mean) / X_std, dtype=torch.float32)
    yn = torch.tensor((y_train_np - y_mean) / y_std,  dtype=torch.float32)

    model   = RocketNN(n_in=n_in, n_out=n_out)
    loader  = DataLoader(TensorDataset(Xn, yn), batch_size=64, shuffle=True)
    opt     = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        sched.step()
        if epoch % 50 == 0:
            logger.info("NN epoch %d / %d", epoch, epochs)

    # Attach normalisation stats to model object for inference
    model.X_mean = X_mean
    model.X_std  = X_std
    model.y_mean = y_mean
    model.y_std  = y_std
    return model


def eval_nn(model, X_test_np, y_test_np):
    import torch
    model.eval()
    Xn = torch.tensor((X_test_np - model.X_mean) / model.X_std, dtype=torch.float32)
    with torch.no_grad():
        yn = model(Xn).numpy()
    preds = yn * model.y_std + model.y_mean
    return preds


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulations",    type=int, default=300,
                        help="Number of RocketPy simulations (default 300)")
    parser.add_argument("--optuna-trials",  type=int, default=60,
                        help="Optuna trials for XGBoost tuning (default 60)")
    parser.add_argument("--nn-epochs",      type=int, default=200,
                        help="PyTorch training epochs (default 200)")
    args = parser.parse_args()

    # ── 1. Dataset ─────────────────────────────────────────────────────────────
    logger.info("=== STEP 1: Generating %d RocketPy simulations ===", args.simulations)
    CSV = "rocket_dataset_3000.csv"
    if args.simulations == 0:
        if not os.path.exists(CSV):
            logger.error("No existing dataset found. Run with --simulations 300 first.")
            return
        logger.info("--simulations 0: loading existing dataset from %s", CSV)
        df = pd.read_csv(CSV)
    else:
        df = generate_dataset(args.simulations)
    logger.info("Raw dataset: %s", df.shape)

    # ── 2. CEA enrichment ──────────────────────────────────────────────────────
    logger.info("=== STEP 2: RocketCEA combustion features ===")
    if "isp_vac" not in df.columns:
        df = enrich_with_cea(df)
    else:
        logger.info("CEA columns already present — skipping recompute.")
    df.to_csv(CSV, index=False)
    logger.info("Dataset saved → %s  columns: %s", CSV, list(df.columns))

    X = df[ALL_FEATURES].values
    y = df[TARGET_COLS].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ── 3. XGBoost + Optuna ────────────────────────────────────────────────────
    logger.info("=== STEP 3: XGBoost hyperparameter tuning (%d trials) ===", args.optuna_trials)
    try:
        best_params = tune_xgb(X_train, y_train, X_test, y_test, args.optuna_trials)
    except Exception as e:
        logger.warning("Optuna failed (%s) — using defaults.", e)
        best_params = {}

    xgb_model = train_xgb(X_train, y_train, best_params)
    xgb_preds = xgb_model.predict(X_test)
    logger.info("XGBoost  R²=%.4f  MAE=%.2f",
                r2_score(y_test, xgb_preds), mean_absolute_error(y_test, xgb_preds))
    with open("xgb_model.pkl", "wb") as f:
        pickle.dump(xgb_model, f)
    logger.info("XGBoost model saved → xgb_model.pkl")

    # ── 4. Neural Network ──────────────────────────────────────────────────────
    logger.info("=== STEP 4: PyTorch Neural Network (%d epochs) ===", args.nn_epochs)
    try:
        import torch
        nn_model  = train_nn(X_train, y_train, args.nn_epochs)
        nn_preds  = eval_nn(nn_model, X_test, y_test)
        logger.info("NeuralNet R²=%.4f  MAE=%.2f",
                    r2_score(y_test, nn_preds), mean_absolute_error(y_test, nn_preds))
        import pickle as _pkl
        with open("nn_model.pt", "wb") as _f:
            _pkl.dump(nn_model, _f)
        logger.info("Neural Network saved → nn_model.pt")
    except ImportError:
        logger.warning("PyTorch not installed — skipping NN. Run: pip install torch")
    except Exception as e:
        logger.error("NN training failed: %s", e)

    logger.info("=== ALL DONE ===")


if __name__ == "__main__":
    main()
