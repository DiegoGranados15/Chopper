import pandas as pd
import json

# Cargar archivos con encoding explícito y manejo de espacios
desc_df = pd.read_csv("data/symptom_description.csv", encoding='utf-8')
prec_df = pd.read_csv("data/symptom_precaution.csv", encoding='utf-8')

# Normalizar nombres: eliminar paréntesis, espacios extra y convertir a minúsculas
def clean_disease_name(name):
    if isinstance(name, str):
        return name.replace("(", "").replace(")", "").strip().lower()
    return name

desc_df["Disease"] = desc_df["Disease"].apply(clean_disease_name)
prec_df["Disease"] = prec_df["Disease"].apply(clean_disease_name)

# Crear la base de datos
disease_db = {}
for _, row in desc_df.iterrows():
    disease = row["Disease"]
    description = row["Description"]
    
    # Buscar precauciones (manejo seguro si no existen)
    precautions = []
    prec_row = prec_df[prec_df["Disease"] == disease]
    
    if not prec_row.empty:
        precautions = [p for p in prec_row.values[0][1:] if pd.notna(p)]
    
    disease_db[disease] = {
        "description": description,
        "precautions": precautions
    }

# Guardar
with open("data/disease_db.json", "w", encoding='utf-8') as f:
    json.dump(disease_db, f, indent=4, ensure_ascii=False)

print("✅ Base de datos creada con éxito!")