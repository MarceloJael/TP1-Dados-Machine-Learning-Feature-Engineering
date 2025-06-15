import pandas as pd
from palmerpenguins import load_penguins

# Carrega o dataset
df = load_penguins()
print("Visualização inicial do dataset:")
print(df.head())

# Mostra as colunas (features)
print("\nFeatures (colunas):")
print(df.columns.tolist())

# Tipo de dados
print("\nTipos de dados:")
print(df.dtypes)