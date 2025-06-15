import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PowerTransformer, StandardScaler
from palmerpenguins import load_penguins

df = load_penguins()
df_clean = df.dropna()

X = df_clean[['flipper_length_mm']].copy()

pt = PowerTransformer(method='yeo-johnson')
X_power = pt.fit_transform(X)

scaler = StandardScaler()
X_zscore = scaler.fit_transform(X)

df_clean['flipper_power'] = X_power
df_clean['flipper_zscore'] = X_zscore

print("== Estatísticas Descritivas ==")
print("\nOriginal:\n", X.describe())
print("\nPowerTransformer:\n", pd.DataFrame(X_power).describe())
print("\nStandardScaler:\n", pd.DataFrame(X_zscore).describe())

plt.figure(figsize=(14, 6))

plt.subplot(2, 3, 1)
X.plot.hist(bins=15, title='Original', legend=False)
plt.xlabel('flipper_length_mm')

plt.subplot(2, 3, 2)
pd.Series(X_power.flatten()).plot.hist(bins=15, title='PowerTransformer', legend=False)
plt.xlabel('Transformado')

plt.subplot(2, 3, 3)
pd.Series(X_zscore.flatten()).plot.hist(bins=15, title='StandardScaler', legend=False)
plt.xlabel('Z-score')

plt.subplot(2, 3, 4)
sns.boxplot(x=X['flipper_length_mm'])
plt.title('Original')

plt.subplot(2, 3, 5)
sns.boxplot(x=X_power.flatten())
plt.title('PowerTransformer')

plt.subplot(2, 3, 6)
sns.boxplot(x=X_zscore.flatten())
plt.title('StandardScaler')

plt.tight_layout()
plt.show()