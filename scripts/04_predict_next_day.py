import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import datetime

# --- CONFIGURATION ---
MODEL_PATH = 'models/final_model.pkl'
FEATURES_PATH = 'data/features_finales.csv' 
REF_DATE = datetime(2025, 12, 3) # <-- VOTRE DATE DE RÉFÉRENCE (Aujourd'hui)
# --- FIN CONFIGURATION ---

# 1. Chargement du Modèle Multi-Sortie
try:
    multi_output_model = joblib.load(MODEL_PATH)
    # Récupérer l'ordre des features à partir du premier estimateur (crucial pour l'input)
    feature_order = multi_output_model.estimators_[0].get_booster().feature_names 
    print(f" Modèle Multi-Sortie chargé depuis : {MODEL_PATH}")
except Exception as e:
    print(f"Erreur de chargement du modèle : {e}")
    sys.exit(1)


# 2. Simulation des Données d'Observation Brutes (J-7 à J)
# Pour une vraie prédiction en 2025, vous devriez appeler l'API Meteostat
# pour les observations des 7 derniers jours.
# Ici, nous SIMULONS cet input en prenant la structure des données d'entraînement.
try:
    df_brut_pour_input = pd.read_csv('data/meteo_brazzaville_daily.csv', index_col='time', parse_dates=True)
except FileNotFoundError:
    print(f"Erreur: Fichier de données brutes introuvable.")
    sys.exit(1)

# Nous prenons les 7 derniers jours de l'historique et les renommons à partir du 04/11/2025
# Cela préserve les tendances Lag (Tmax_J-1, Tmax_J-2, etc.) mais utilise la bonne saisonnalité.
df_7_jours_simules = df_brut_pour_input.iloc[-7:].copy() # Les 7 derniers jours d'observation de l'historique (2020)
df_7_jours_simules.index = pd.to_datetime(pd.date_range(end=REF_DATE, periods=7))


# --- 3. FONCTION DE CRÉATION DE FEATURES ---
def creer_features_pour_prediction(df_historique_7_jours, date_cible):
    df_temp = df_historique_7_jours.copy()
    
    # Création des Lag Features
    lags = [1, 2, 3, 7]
    for lag in lags:
        df_temp[f'Tmax_Lag_{lag}'] = df_temp['temperature_max_jour'].shift(lag)
        df_temp[f'Tmin_Lag_{lag}'] = df_temp['temperature_min_jour'].shift(lag)
        df_temp[f'Prcp_Lag_{lag}'] = df_temp['precipitation_somme_jour'].shift(lag)

    # Création des Features Temporelles pour la date cible (celle qui sera J)
    df_temp['Mois'] = date_cible.month
    df_temp['Jour_de_Annee'] = date_cible.timetuple().tm_yday
    df_temp['Jour_de_Semaine'] = date_cible.weekday()
    
    # Isoler le dernier jour, qui est le seul qui aura toutes ses Lag Features complètes
    X_pred_raw = df_temp.iloc[-1].to_frame().T
    X_pred_raw.index = [date_cible]
    
    # Nettoyer les colonnes brutes
    colonnes_brutes = ['temperature_max_jour', 'temperature_min_jour', 
                      'precipitation_somme_jour', 'vitesse_vent_moyenne_jour']
    X_pred = X_pred_raw.drop(columns=colonnes_brutes, errors='ignore')
    
    # Assurer l'ordre des colonnes
    X_pred = X_pred[feature_order]
    
    return X_pred


# --- 4. PRÉDICTION J+1 (Demain : 2025-11-11) ---

date_pred_j1 = REF_DATE + pd.Timedelta(days=1)
X_pred_j1 = creer_features_pour_prediction(df_7_jours_simules, REF_DATE)

predictions_j1 = multi_output_model.predict(X_pred_j1.values)[0]
tmax_j1 = predictions_j1[0]
tmin_j1 = predictions_j1[1]


# --- 5. PRÉDICTION J+2 (Après-demain : 2025-11-12) : Prédiction Récursive ---

date_pred_j2 = REF_DATE + pd.Timedelta(days=2)

# Étape 1 : Créer la nouvelle ligne d'observation (J+1) en utilisant la prédiction J+1
# Pour la précipitation, nous supposons la valeur du dernier jour connu (J=10/11/2025)
prcp_j1_assumee = df_7_jours_simules['precipitation_somme_jour'].iloc[-1]
wspd_j1_assume = df_7_jours_simules['vitesse_vent_moyenne_jour'].iloc[-1]

nouvelle_observation_j1 = pd.DataFrame({
    'temperature_max_jour': [tmax_j1], 
    'temperature_min_jour': [tmin_j1],
    'precipitation_somme_jour': [prcp_j1_assumee],
    'vitesse_vent_moyenne_jour': [wspd_j1_assume]
}, index=[date_pred_j1])

# Étape 2 : Ajouter la prédiction J+1 à l'historique simulé pour prédire J+2
df_8_jours_pour_j2 = pd.concat([df_7_jours_simules, nouvelle_observation_j1])

# Étape 3 : Créer les features pour la prédiction J+2 (J+1 est maintenant le J-1)
X_pred_j2 = creer_features_pour_prediction(df_8_jours_pour_j2, date_pred_j1)

predictions_j2 = multi_output_model.predict(X_pred_j2.values)[0]
tmax_j2 = predictions_j2[0]
tmin_j2 = predictions_j2[1]


# 6. AFFICHAGE DES RÉSULTATS
print("\n---------------------------------------------------------")
print(f"   Données de référence : {REF_DATE.strftime('%Y-%m-%d')} (Aujourd'hui)")
print("---------------------------------------------------------")
print(f"   Jour de la Prédiction (J+1) : {date_pred_j1.strftime('%Y-%m-%d')} (Demain)")
print(f"   Prédiction Température MAX : {tmax_j1:.2f} °C")
print(f"   Prédiction Température MIN : {tmin_j1:.2f} °C")
print("\n")
print(f"   Jour de la Prédiction (J+2) : {date_pred_j2.strftime('%Y-%m-%d')} (Après-demain)")
print(f"   Prédiction Température MAX : {tmax_j2:.2f} °C")
print(f"   Prédiction Température MIN : {tmin_j2:.2f} °C")
print("---------------------------------------------------------")
