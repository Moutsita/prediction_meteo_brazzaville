import pandas as pd
import os
import sys
import numpy as np

# --- CONFIGURATION ---
INPUT_PATH = 'data/meteo_brazzaville_daily.csv'
OUTPUT_PATH = 'data/features_finales.csv'
# --- FIN CONFIGURATION ---

try:
    df = pd.read_csv(INPUT_PATH, index_col='time', parse_dates=True)
except FileNotFoundError:
    print(f"Erreur: Le fichier d'entrée {INPUT_PATH} est introuvable. Exécutez le script 01 en premier.")
    sys.exit(1)

print(f"Chargement des données brutes réussi. Taille initiale: {df.shape}")

# --- ÉTAPE 1 : CRÉATION DES VARIABLES CIBLES (Y) : T° Max et T° Min J+1 ---
# Shift de -1 place la T° Max/Min du lendemain sur la ligne du jour J
df['Tmax_Demain'] = df['temperature_max_jour'].shift(-1)
df['Tmin_Demain'] = df['temperature_min_jour'].shift(-1)

# --- ÉTAPE 2 : CRÉATION DES FEATURES TEMPÉRELLES ---
df['Mois'] = df.index.month
df['Jour_de_Annee'] = df.index.dayofyear
df['Jour_de_Semaine'] = df.index.dayofweek

# --- ÉTAPE 3 : CRÉATION DES FEATURES DE DÉCALAGE (LAG FEATURES) ---
lags = [1, 2, 3, 7] # J-1, J-2, J-3, J-7
for lag in lags:
    df[f'Tmax_Lag_{lag}'] = df['temperature_max_jour'].shift(lag)
    df[f'Tmin_Lag_{lag}'] = df['temperature_min_jour'].shift(lag)
    df[f'Prcp_Lag_{lag}'] = df['precipitation_somme_jour'].shift(lag)


# --- ÉTAPE 4 : NETTOYAGE FINAL ---
df = df.dropna()

# 2. Définition des Features (X) et des Cibles (Y)
colonnes_cibles = ['Tmax_Demain', 'Tmin_Demain'] # <-- LES DEUX CIBLES
Y = df[colonnes_cibles]
# X : on retire les variables brutes de J0 et les cibles
colonnes_a_retirer = colonnes_cibles + ['temperature_max_jour', 'temperature_min_jour', 
                      'precipitation_somme_jour', 'vitesse_vent_moyenne_jour']
X = df.drop(columns=colonnes_a_retirer, errors='ignore')

# --- ÉTAPE 5 : SAUVEGARDE ---
df_final = X.join(Y)
df_final.to_csv(OUTPUT_PATH)

print(f"\nFichier de features {OUTPUT_PATH} créé avec succès.")
print(f"Nombre de jours utilisables après nettoyage: {df_final.shape[0]}")
print(f"Nombre de features créées (X): {X.shape[1]}")