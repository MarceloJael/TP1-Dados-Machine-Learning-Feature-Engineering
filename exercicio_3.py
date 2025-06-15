import pandas as pd
import matplotlib.pyplot as plt
from palmerpenguins import load_penguins

df = load_penguins()
df_clean = df.dropna()

massa = df_clean['body_mass_g']

bins = [2700, 3300, 3900, 4500, 5100, 5700]
labels = ['Muito leve', 'Leve', 'Médio', 'Pesado', 'Muito pesado']

df_clean['massa_discretizada'] = pd.cut(massa, bins=bins, labels=labels, include_lowest=True)

print(df_clean[['body_mass_g', 'massa_discretizada']].head(10))

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(massa, bins=10)
plt.title('Distribuição Original da Massa Corporal')
plt.xlabel('body_mass_g')

plt.subplot(1, 2, 2)
df_clean['massa_discretizada'].value_counts().sort_index().plot(kind='bar')
plt.title('Distribuição por Faixa (Discretização)')
plt.xlabel('Categoria')
plt.ylabel('Número de Pinguins')

plt.tight_layout()
plt.show()