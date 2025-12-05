import streamlit as st
import pandas as pd
import numpy as np
import joblib
# Import et ajustement des librairies datetime :
from datetime import datetime, timedelta 
import os
import sys

# Import de Meteostat pour l'appel API en temps réel
from meteostat import Point, Daily 

# --- CONFIGURATION DU PROJET ---
MODEL_PATH = 'models/final_model.pkl'
DATA_PATH = 'data/meteo_brazzaville_daily.csv'
BRAZZAVILLE_STATION_ID = '64450' 
MODEL_MAE = 1.32 

# COORDONNÉES DE BRAZZAVILLE (Station ID 64450)
BRAZZAVILLE_LAT = -4.25
BRAZZAVILLE_LON = 15.25
# --- FIN CONFIGURATION ---

# 1. Configuration de l'interface Streamlit
st.set_page_config(page_title="Prévision Météo Brazzaville (J+2) - Master IA", layout="wide")
st.title("Prévisions Météo Brazzaville et Analyse Climatique")
st.markdown("---")


# --- FONCTIONS CLÉS ---

@st.cache_resource
def load_resources():
    """Charge le modèle, les données historiques, et calcule les normales climatiques (1991-2020)."""
    try:
        multi_output_model = joblib.load(MODEL_PATH)
        if not multi_output_model.estimators_ or not multi_output_model.estimators_[0].get_booster().feature_names:
             raise ValueError("Le modèle chargé n'a pas les attributs d'estimateur ou de noms de features attendus.")
        
        feature_order = multi_output_model.estimators_[0].get_booster().feature_names
        
        # Charger les données brutes pour le calcul des normales
        df_brut = pd.read_csv(DATA_PATH, index_col='time', parse_dates=True)
        
        # --- CALCUL DES NORMALES CLIMATIQUES (Moyenne 1991-2020) ---
        df_normales = df_brut.copy()
        df_normales['Jour_de_Annee'] = df_normales.index.dayofyear
        
        normales_journalieres = df_normales.groupby('Jour_de_Annee').agg(
            Tmax_Normale=('temperature_max_jour', 'mean'),
            Tmin_Normale=('temperature_min_jour', 'mean')
        ).reset_index()
        
        return multi_output_model, feature_order, normales_journalieres
    except Exception as e:
        st.error(f"Erreur de chargement des ressources (modèle/normales). Assurez-vous que les fichiers existent.")
        st.exception(e)
        st.stop()
        
def get_real_time_input(date_ref):
    """
    Récupère les 7 jours d'observations réelles précédant la date de référence (J-7 à J-1) via API.
    """
    with st.spinner(f"Connexion à Meteostat (Station {BRAZZAVILLE_STATION_ID}) pour les observations récentes..."):
        
        # CORRECTION TYPERROR : Convertir l'objet date (de st.date_input) en datetime (minuit)
        date_ref_dt = datetime.combine(date_ref, datetime.min.time())

        # Les variables start et end sont maintenant des objets datetime.datetime
        start = date_ref_dt - timedelta(days=7)
        end = date_ref_dt - timedelta(days=1)
        
        location = Point(BRAZZAVILLE_LAT, BRAZZAVILLE_LON) 
        
        data = Daily(location, start, end)
        data = data.fetch()
        
        # Renommer les colonnes
        data = data.rename(columns={
            'tmax': 'temperature_max_jour',
            'tmin': 'temperature_min_jour',
            'prcp': 'precipitation_somme_jour',
            'wspd': 'vitesse_vent_moyenne_jour'
        })
        
        # CORRECTION KEYERROR : Utiliser le nom de colonne renommé
        data = data[['temperature_max_jour', 'temperature_min_jour', 
                     'precipitation_somme_jour', 'vitesse_vent_moyenne_jour']]
        
        data = data.sort_index()
        
        return data

def creer_features_pour_prediction(df_historique_7_jours, date_cible, feature_order):
    """Crée les features Lag et Temporelles pour la date cible."""
    df_temp = df_historique_7_jours.copy()
    
    # Création des Lag Features
    lags = [1, 2, 3, 7]
    for lag in lags:
        df_temp[f'Tmax_Lag_{lag}'] = df_temp['temperature_max_jour'].shift(lag)
        df_temp[f'Tmin_Lag_{lag}'] = df_temp['temperature_min_jour'].shift(lag)
        df_temp[f'Prcp_Lag_{lag}'] = df_temp['precipitation_somme_jour'].shift(lag)

    # Création des Features Temporelles pour la date cible
    df_temp['Mois'] = date_cible.month
    df_temp['Jour_de_Annee'] = date_cible.timetuple().tm_yday
    df_temp['Jour_de_Semaine'] = date_cible.weekday()
    
    # Isoler le dernier jour (qui est la date cible)
    X_pred_raw = df_temp.iloc[-1].to_frame().T
    
    # Nettoyer et Assurer l'ordre des colonnes
    colonnes_brutes = ['temperature_max_jour', 'temperature_min_jour', 
                      'precipitation_somme_jour', 'vitesse_vent_moyenne_jour']
    X_pred = X_pred_raw.drop(columns=colonnes_brutes, errors='ignore')
    
    try:
        X_pred = X_pred[feature_order]
    except KeyError as e:
        st.error(f"Erreur Fatale : Une feature (colonne) est manquante pour l'input du modèle. {e}")
        return None

    return X_pred


# --- LOGIQUE PRINCIPALE ---
multi_output_model, feature_order, normales_journalieres = load_resources()

# 2. Sélecteur de Date (J) par l'utilisateur
st.sidebar.header("Choisir la Date de Référence (J)")
st.sidebar.markdown("Le modèle prédira pour J+1 et J+2 (temps réel).")
max_date_selectable = datetime.now().date() + timedelta(days=1)
min_date_selectable = max_date_selectable - timedelta(days=100) 

REF_DATE = st.sidebar.date_input(
    "Date de Prévision (J) :",
    value=datetime.now().date(), 
    min_value=min_date_selectable, 
    max_value=max_date_selectable
)

# 3. Récupération des données d'input réelles
df_observations_reelles = get_real_time_input(REF_DATE) 

if df_observations_reelles is None or len(df_observations_reelles) < 7:
    st.error(f"**Données Insuffisantes :** L'API n'a pas pu fournir les 7 jours d'observations (J-7 à J-1) pour la date choisie ({REF_DATE.strftime('%d %B %Y')}).")
    st.markdown("Veuillez choisir une date plus ancienne ou vérifier la connexion internet/disponibilité des données de la station.")
    st.stop()
    
st.success(f"7 jours d'observations réelles chargés avec succès (du {(REF_DATE - timedelta(days=7)).strftime('%d %B %Y')} au {(REF_DATE - timedelta(days=1)).strftime('%d %B %Y')}).")
st.markdown("---")


# --- BOUTON DE PRÉDICTION ---
if st.button("Lancer la Prévision pour J+1 et J+2", type="primary"):
    
    with st.spinner(f'Calcul des prévisions pour le {REF_DATE.strftime("%d/%m")} en cours...'):
        
        # --- PRÉDICTION J+1 (Demain) ---
        date_pred_j1 = REF_DATE + timedelta(days=1)
        
        # 1. Préparation de l'input pour J+1:
        df_input_pour_j1 = pd.concat([df_observations_reelles, pd.DataFrame(index=[REF_DATE])])
        
        # 2. Création des features et prédiction J+1 (cible temporelle: J)
        X_pred_j1 = creer_features_pour_prediction(df_input_pour_j1, REF_DATE, feature_order)
        
        predictions_j1 = multi_output_model.predict(X_pred_j1.values)[0]
        tmax_j1 = predictions_j1[0]
        tmin_j1 = predictions_j1[1]

        
        # --- PRÉDICTION J+2 (Après-demain) : Prédiction Récursive ---
        date_pred_j2 = REF_DATE + timedelta(days=2)
        
        # 1. Créer la ligne d'observation (J+1) en utilisant la prédiction J+1 (tmax_j1/tmin_j1)
        prcp_j_assumee = df_observations_reelles['precipitation_somme_jour'].iloc[-1]
        wspd_j_assume = df_observations_reelles['vitesse_vent_moyenne_jour'].iloc[-1]

        nouvelle_observation_j_plus_1 = pd.DataFrame({
            'temperature_max_jour': [tmax_j1], 
            'temperature_min_jour': [tmin_j1],
            'precipitation_somme_jour': [prcp_j_assumee],
            'vitesse_vent_moyenne_jour': [wspd_j_assume]
        }, index=[date_pred_j1]) 

        # 2. History pour J+2: Utiliser les 6 dernières observations réelles (J-6 à J-1) + la prédiction J+1
        df_j2_history_base = df_observations_reelles.iloc[1:] 
        
        # On ajoute la prédiction J+1 pour créer l'historique de Lag
        df_j2_history_complete = pd.concat([df_j2_history_base, nouvelle_observation_j_plus_1])
        
        # 3. Ajouter la ligne vide pour J+2 (cible)
        df_input_j2_final = pd.concat([df_j2_history_complete, pd.DataFrame(index=[date_pred_j2])])
        
        # 4. Création des features J+2 (cible temporelle: J+2)
        X_pred_j2 = creer_features_pour_prediction(df_input_j2_final, date_pred_j2, feature_order)

        predictions_j2 = multi_output_model.predict(X_pred_j2.values)[0]
        tmax_j2 = predictions_j2[0]
        tmin_j2 = predictions_j2[1]
        
    
    # --- AFFICHAGE DES RÉSULTATS + ANALYSE CLIMATIQUE ---
    st.success("Prévisions générées avec succès!")

    col1, col2 = st.columns(2)
    
    for date_pred, tmax_pred, tmin_pred in [(date_pred_j1, tmax_j1, tmin_j1), (date_pred_j2, tmax_j2, tmin_j2)]:
        
        jour_de_annee = date_pred.timetuple().tm_yday
        # Récupération de la normale (1991-2020) pour ce jour de l'année
        normale_du_jour = normales_journalieres[normales_journalieres['Jour_de_Annee'] == jour_de_annee].iloc[0]
        tmax_normale = normale_du_jour['Tmax_Normale']
        tmin_normale = normale_du_jour['Tmin_Normale']

        # Calculer l'écart
        ecart_tmax = tmax_pred - tmax_normale
        ecart_tmin = tmin_pred - tmin_normale
        
        col = col1 if date_pred == date_pred_j1 else col2
        
        with col:
            st.header(f"Prévision pour le :")
            st.subheader(date_pred.strftime('%d %B %Y'))

            st.markdown(f"**T. MAX Prédite :** `{tmax_pred:.2f} °C`")
            st.metric(
                "Écart vs. Normale (Tmax)", 
                f"{ecart_tmax:.2f} °C", 
                delta=f"(Normale 1991-2020 : {tmax_normale:.2f} °C)",
                delta_color="inverse" if ecart_tmax < 0 else "normal"
            )

            st.markdown(f"**T. MIN Prédite :** `{tmin_pred:.2f} °C`")
            st.metric(
                "Écart vs. Normale (Tmin)", 
                f"{ecart_tmin:.2f} °C", 
                delta=f"(Normale 1991-2020 : {tmin_normale:.2f} °C)",
                delta_color="inverse" if ecart_tmin < 0 else "normal"
            )
            st.markdown("---")
            
    st.caption(f"Le modèle (MAE $\\approx$ {MODEL_MAE:.2f} °C) utilise les observations en temps réel de la station {BRAZZAVILLE_STATION_ID} pour prédire.")