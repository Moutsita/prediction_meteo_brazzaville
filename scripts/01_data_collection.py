import pandas as pd
from meteostat import Daily
from datetime import datetime
import os
import sys

# --- CONFIGURATION ---
STATION_ID = '64450' # Brazzaville
DATE_DEBUT = datetime(1991, 1, 1)
DATE_FIN = datetime(2020, 12, 31)
FILE_PATH = 'data/meteo_brazzaville_daily.csv'
# --- FIN CONFIGURATION ---

# Assurer que le dossier 'data' existe
if not os.path.exists('data'):
    os.makedirs('data')

print("Recherche et téléchargement des données d'observation (Meteostat) en cours...")

try:
    # 1. Requête et Récupération des données quotidiennes
    data = Daily(STATION_ID, DATE_DEBUT, DATE_FIN)
    data = data.fetch()

    if data.empty:
        print("Erreur: Aucune donnée d'observation disponible pour la station 64450 pour cette période.")
        sys.exit(1) # Quitter avec un code d'erreur

    # 2. Renommage et nettoyage initial
    df = data.rename(columns={
        'tmax': 'temperature_max_jour',
        'tmin': 'temperature_min_jour',
        'prcp': 'precipitation_somme_jour',
        'wspd': 'vitesse_vent_moyenne_jour'
    })
    
    df = df[['temperature_max_jour', 'temperature_min_jour', 'precipitation_somme_jour', 'vitesse_vent_moyenne_jour']]
    
    # 3. Traitement des valeurs manquantes (imputation par la moyenne pour les features)
    df = df.fillna(df.mean()) 
    
    # 4. SAUVEGARDE des données
    df.to_csv(FILE_PATH)
    print(f"\nFichier {FILE_PATH} créé avec succès. Dimensions: {df.shape}")

except Exception as e:
    print(f"Une erreur est survenue lors de la collecte de données : {e}")
    sys.exit(1)