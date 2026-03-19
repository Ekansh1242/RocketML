# 🚀 Rocket Performance Prediction API  —  v2.0

A Flask REST API predicting rocket flight performance using **XGBoost** (Optuna-tuned) and a **PyTorch Neural Network**, both trained on [RocketPy](https://rocketpy.org/) flight simulations enriched with **RocketCEA** combustion physics.

---

## What's new in v2

| | v1 | v2 |
|---|---|---|
| ML model | Random Forest | **XGBoost** (Optuna-tuned) |
| Deep model | ✗ | **PyTorch MLP** (256→256→128) |
| Combustion physics | ✗ | **RocketCEA** (Isp, c*, Cf, Pc) |
| Hyperparameter tuning | Manual | **Optuna Bayesian search** |
| Runtime model switch | ✗ | `/model/switch` endpoint |

---

## Architecture

```
RocketPy simulation
       ↓
  Raw features (8)          ← thrust, burn_time, mass, drag, angle,
       +                       diameter, propellant_mass, wind
RocketCEA combustion (4)    ← Isp_vac, c*, Cf, chamber_pressure
       ↓
  12 features total
       ↓
 ┌─────────────┐   ┌──────────────────────────────┐
 │  XGBoost    │   │  PyTorch MLP                 │
 │  (default)  │   │  256 → 256 → 128 → 2 outputs │
 └─────────────┘   └──────────────────────────────┘
       ↓
  apogee (m)  +  max_velocity (m/s)
```

---

## Quick Start

```bash
git clone https://github.com/<your-username>/rocket-ml-api.git
cd rocket-ml-api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Generate dataset, tune XGBoost with Optuna, train PyTorch NN
python generate_dataset.py --simulations 3000 --optuna-trials 60 --nn-epochs 200

# Start API
python app.py
```

> **Quick test:** use `--simulations 300 --optuna-trials 20` first.

---

## API Reference

### `POST /predict`
```json
{
  "thrust": 1500, "burn_time": 5.0, "mass": 70, "drag": 0.4,
  "angle": 85, "diameter": 0.2, "propellant_mass": 25, "wind": 3.0
}
```
Response includes both design inputs **and** the auto-computed combustion parameters:
```json
{
  "model": "xgb",
  "combustion": { "isp_vac": 262.5, "c_star": 1545.0, "cf": 1.668, "chamber_pressure_bar": 37.5 },
  "predictions": { "apogee_m": 3241.87, "max_velocity_ms": 198.45 }
}
```

### `POST /optimize`
```json
{ "n_trials": 120 }
```
Runs Optuna over all 8 design parameters using the active model.

### `POST /monte_carlo`
```json
{ ...design params..., "n_samples": 300, "noise_std": 0.05 }
```

### `GET /combustion?thrust=1500&burn_time=5&propellant_mass=25`
Returns CEA combustion parameters for a given engine config.

### `POST /model/switch`
```json
{ "model": "xgb" }   // or "nn"
```
Switch between XGBoost and the PyTorch Neural Network at runtime.

### `GET /feature_importance`
Returns importance of all 12 features (design + combustion) from XGBoost.

---

## Combustion Features (RocketCEA)

| Feature | Description | Typical range |
|---|---|---|
| `isp_vac` | Vacuum specific impulse (s) | 250–280 s |
| `c_star` | Characteristic exhaust velocity (m/s) | 1450–1620 m/s |
| `cf` | Thrust coefficient | 1.60–1.75 |
| `chamber_pressure_bar` | Chamber pressure (bar) | 20–80 bar |

If `rocketcea` is not installed, physics-informed approximations (Sutton & Biblarz) are used automatically.

---

## Input Ranges

| Feature | Min | Max | Unit |
|---|---|---|---|
| thrust | 800 | 2000 | N |
| burn_time | 3 | 8 | s |
| mass | 40 | 100 | kg |
| drag | 0.2 | 0.6 | – |
| angle | 75 | 90 | ° |
| diameter | 0.1 | 0.3 | m |
| propellant_mass | 10 | 40 | kg |
| wind | 0 | 10 | m/s |

---

## Project Structure

```
rocket-ml-api/
├── app.py                  # Flask API (XGBoost + NN + CEA)
├── generate_dataset.py     # Simulation → CEA → XGBoost tune → NN train
├── requirements.txt
├── .gitignore
├── Procfile
└── README.md
```

---

## Deployment

```bash
# Heroku / Render — Procfile already configured
gunicorn -w 4 -b 0.0.0.0:$PORT app:app
```

### Docker
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

---

## Known Limitations

- RocketPy models **flight trajectory**, not internal combustion thermodynamics
- CEA combustion features assume **HTPB/AP composite** propellant (O/F = 2.3)
- XGBoost and NN are **surrogate models** — not physics simulators
- For true combustion simulation, integrate a full CEA pipeline with real propellant chemistry

---

## License

MIT
