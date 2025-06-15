import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from palmerpenguins import load_penguins

df = load_penguins()
df_clean = df.dropna()

variaveis = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
dados_numericos = df_clean[variaveis]

scaler = MinMaxScaler()
dados_normalizados = scaler.fit_transform(dados_numericos)

df_normalizado = pd.DataFrame(dados_normalizados, columns=[f"{col}_minmax" for col in variaveis])

df_resultado = pd.concat([dados_numericos.reset_index(drop=True), df_normalizado], axis=1)

print(df_resultado.head())

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
df_resultado['body_mass_g'].plot.hist(bins=20, title='Original - body_mass_g')
plt.xlabel('body_mass_g')

plt.subplot(1, 2, 2)
df_resultado['body_mass_g_minmax'].plot.hist(bins=20, title='Min-Max Normalizado')
plt.xlabel('body_mass_g_minmax')

plt.tight_layout()
plt.show()