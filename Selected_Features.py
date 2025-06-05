import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# 1) Load the training data
df_train = pd.read_csv("feature_matrix_train.csv")
feature_names = df_train.columns[3:]  # drop SampleID, Variant, Global CFR
X_train = df_train[feature_names].values
y_train = df_train["Global CFR"].values

# 2) Scale the features (binary data)
scaler = StandardScaler(with_mean=False)
X_train_scaled = scaler.fit_transform(X_train)

# 3) Fit Lasso with the chosen alpha
alpha = 0.000281
lasso = Lasso(alpha=alpha, max_iter=10000)
lasso.fit(X_train_scaled, y_train)

# 4) Extract and print the selected features
coefs = lasso.coef_
selected_idx = np.where(coefs != 0)[0]

print(f"Features selected at α={alpha}:")
for idx in selected_idx:
    print(f"  {feature_names[idx]}  →  coefficient = {coefs[idx]:.6f}")
