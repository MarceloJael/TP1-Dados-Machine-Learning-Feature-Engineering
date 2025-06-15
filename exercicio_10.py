import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import FunctionTransformer
from palmerpenguins import load_penguins

df = load_penguins()
df_clean = df.dropna()

X = df_clean[['flipper_length_mm']].copy()

log_transformer = FunctionTransformer(np.log10, validate=True)
X_log = log_transformer.fit_transform(X)

exp_inverse_transformer = FunctionTransformer(lambda x: np.power(10, x), validate=True)
X_log_exp = exp_inverse_transformer.fit_transform(X_log)

df_clean['flipper_log10'] = X_log
df_clean['flipper_log10_exp_inverse'] = X_log_exp

print("Original:\n", X.describe(), "\n")
print("Após log10:\n", df_clean['flipper_log10'].describe(), "\n")
print("Após log10 + exponencial inversa:\n", df_clean['flipper_log10_exp_inverse'].describe())

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
X.plot.hist(bins=15, title='Original', legend=False)
plt.xlabel('flipper_length_mm')

plt.subplot(1, 3, 2)
df_clean['flipper_log10'].plot.hist(bins=15, title='Log10 Transformado', legend=False)
plt.xlabel('log10(flipper_length_mm)')

plt.subplot(1, 3, 3)
df_clean['flipper_log10_exp_inverse'].plot.hist(bins=15, title='Após Log + Inversa', legend=False)
plt.xlabel('Valor reconstruído')

plt.tight_layout()
plt.show()