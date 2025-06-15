import pandas as pd
import matplotlib.pyplot as plt
from palmerpenguins import load_penguins

df = load_penguins()
df_clean = df.dropna()

massa = df_clean['body_mass_g']

df_clean['massa_quantis'] = pd.qcut(massa, q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

print(df_clean[['body_mass_g', 'massa_quantis']].head(10))

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
massa.plot.hist(bins=10, title='Distribuição Original da Massa Corporal')
plt.xlabel('body_mass_g')

plt.subplot(1, 2, 2)
df_clean['massa_quantis'].value_counts().sort_index().plot(kind='bar')
plt.title('Distribuição por Quantis (Bins Variáveis)')
plt.xlabel('Quantil')
plt.ylabel('Número de Pinguins')

plt.tight_layout()
plt.show()