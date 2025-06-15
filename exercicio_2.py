from palmerpenguins import load_penguins

df = load_penguins()
df_clean = df.dropna()

bill_lengths = df_clean['bill_length_mm']
body_masses = df_clean['body_mass_g']

escalar_exemplo = bill_lengths.iloc[0]
print(f"Tamanho de escalar: {escalar_exemplo}")

vetor_exemplo = df_clean[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']].iloc[0]
print("\nVetor:")
print(vetor_exemplo)

espaco_vetorial = df_clean[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
print("\nEspaço vetorial:", espaco_vetorial.shape)