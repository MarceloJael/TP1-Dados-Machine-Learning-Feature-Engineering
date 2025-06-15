import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from palmerpenguins import load_penguins

df = load_penguins()
df_clean = df.dropna()

X_original = df_clean[['bill_length_mm']].copy()

scaler = MinMaxScaler()
X_minmax = scaler.fit_transform(X_original)

df_clean['bill_length_mm_minmax'] = X_minmax

print("Original:\n", X_original.describe(), "\n")
print("Normalizado (Min-Max):\n", pd.DataFrame(X_minmax, columns=['bill_length_mm_minmax']).describe())

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
X_original.plot.hist(bins=15, title='Original - bill_length_mm', legend=False)
plt.xlabel('bill_length_mm')

plt.subplot(1, 2, 2)
df_clean['bill_length_mm_minmax'].plot.hist(bins=15, title='Normalizado - Min-Max', legend=False)
plt.xlabel('bill_length_mm_minmax')

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(df_clean[['bill_length_mm', 'bill_length_mm_minmax']].head(20), annot=True, cmap='YlGnBu')
plt.title('Comparação Original vs Min-Max (20 primeiras linhas)')
plt.show()