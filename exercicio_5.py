import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import FunctionTransformer
from palmerpenguins import load_penguins

df = load_penguins()
df_clean = df.dropna()

flipper = df_clean[['flipper_length_mm']]

log_transformer = FunctionTransformer(np.log10, validate=True)

flipper_log = log_transformer.transform(flipper)

df_clean['flipper_log10'] = flipper_log

print("\nEstatísticas - Antes:")
print(flipper.describe())
print("\nEstatísticas - Depois (log10):")
print(df_clean['flipper_log10'].describe())

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
flipper.plot.hist(bins=20, title='Original - flipper_length_mm', legend=False)
plt.xlabel('flipper_length_mm')

plt.subplot(1, 2, 2)
df_clean['flipper_log10'].plot.hist(bins=20, title='Log10 Transformado', legend=False)
plt.xlabel('log10(flipper_length_mm)')

plt.tight_layout()
plt.show()