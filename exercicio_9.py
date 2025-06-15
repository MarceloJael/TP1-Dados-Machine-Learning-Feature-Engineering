import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from palmerpenguins import load_penguins

df = load_penguins()
df_clean = df.dropna()

features = df_clean[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm']]
target = df_clean['body_mass_g']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)

y_pred = ridge.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")
print("\nCoeficientes do modelo:")
print(pd.Series(ridge.coef_, index=features.columns))

plt.scatter(y_test, y_pred)
plt.xlabel('Massa Real (g)')
plt.ylabel('Massa Prevista (g)')
plt.title('Ridge Regression: Massa Corporal')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.grid(True)
plt.tight_layout()
plt.show()