import pandas as pd

# 1. Chargement (adapte le nom du fichier)
# Si ton fichier utilise des virgules comme séparateurs
df = pd.read_csv('data/export.csv') 

# 2. Conversion de la date
# On transforme le texte "2023-11-01..." en objet date compréhensible par Python
df['horodate'] = pd.to_datetime(df['horodate'], utc=True)

# 3. Création des variables temporelles (Feature Engineering)
df['heure'] = df['horodate'].dt.hour
df['jour_nom'] = df['horodate'].dt.day_name()
df['est_weekend'] = df['horodate'].dt.dayofweek >= 5 # Samedi=5, Dimanche=6

# 4. Aperçu
print(df.head())
print(f"\nNombre de lignes : {len(df)}")