import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Load pre‑scaled data (or scale here as before)
X_train = pd.read_csv("feature_matrix_train.csv").drop(
    ["SampleID","Variant","Global CFR"], axis=1
).values
y_train = pd.read_csv("feature_matrix_train.csv")["Global CFR"].values
X_test = pd.read_csv("feature_matrix_test.csv").drop(
    ["SampleID","Variant","Global CFR"], axis=1
).values
y_test = pd.read_csv("feature_matrix_test.csv")["Global CFR"].values

scaler = StandardScaler(with_mean=False)
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# 1) Define a grid of alphas to try (log‐spaced from 1e-5 to 1e-2)
alphas = np.logspace(-5, -2, 30)

results = []
for a in alphas:
    l = Lasso(alpha=a, max_iter=10000)
    l.fit(X_train_s, y_train)
    n_feats = np.sum(l.coef_ != 0)
    r2 = r2_score(y_test, l.predict(X_test_s))
    results.append((a, n_feats, r2))

# 2) Put into a DataFrame for easy inspection
df = pd.DataFrame(results, columns=["alpha","n_features","test_R2"])
print(df)
