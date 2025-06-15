import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer
from palmerpenguins import load_penguins

df = load_penguins()
df_clean = df.dropna()

massa = df_clean[['body_mass_g']]

power = PowerTransformer(method='yeo-johnson', standardize=True)
massa_power = power.fit_transform(massa)

df_clean['massa_power'] = massa_power

print("\nEstatísticas - Original:")
print(massa.describe())
print("\nEstatísticas - PowerTransformada:")
print(df_clean['massa_power'].describe())

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
massa.plot.hist(bins=20, title='Original - body_mass_g', legend=False)
plt.xlabel('body_mass_g')

plt.subplot(1, 2, 2)
df_clean['massa_power'].plot.hist(bins=20, title='PowerTransformer', legend=False)
plt.xlabel('Transformado')

plt.tight_layout()
plt.show()