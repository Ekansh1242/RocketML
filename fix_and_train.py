import pandas as pd, pickle, torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

ALL_FEATURES = ['thrust','burn_time','mass','drag','angle','diameter',
                'propellant_mass','wind','isp_vac','c_star','cf','chamber_pressure_bar']
TARGET_COLS  = ['apogee','velocity']

# Step 1 — Add CEA columns only if missing
df = pd.read_csv('rocket_dataset_3000.csv')

if 'isp_vac' not in df.columns:
    def cea(row):
        pc = max(20, min(80, row.thrust / (row.burn_time * 200)))
        return pd.Series({
            'isp_vac': 250 + (pc-20)*0.5,
            'c_star':  1500 + (pc-20)*3.0,
            'cf':      1.65 + (pc-20)*0.002,
            'chamber_pressure_bar': pc
        })
    df = df.join(df.apply(cea, axis=1))
    df.to_csv('rocket_dataset_3000.csv', index=False)
    print("CEA columns added. Columns:", list(df.columns))
else:
    print("CEA columns already present:", [c for c in df.columns if c in ALL_FEATURES])

# Step 2 — Train Neural Network
from generate_dataset_v2 import train_nn, eval_nn

X = df[ALL_FEATURES].values
y = df[TARGET_COLS].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nn = train_nn(X_train, y_train, 200)
preds = eval_nn(nn, X_test, y_test)
print("NN R2:", r2_score(y_test, preds))
torch.save(nn, 'nn_model.pt')
print("Done — nn_model.pt saved")
