# Fichier : scripts/05_analysis_and_visualization.py

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
import os
import sys

# --- CONFIGURATION ---
INPUT_PATH = 'data/features_finales.csv'
MODEL_PATH = 'models/final_model.pkl'
TEST_SPLIT_DATE = '2018-01-01'
# --- FIN CONFIGURATION ---

# 1. Chargement des données et du modèle
try:
    df = pd.read_csv(INPUT_PATH, index_col='time', parse_dates=True)
    multi_output_model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    print("Erreur: Assurez-vous que les scripts 02 et 03 ont été exécutés avec succès.")
    sys.exit(1)

# Séparation des cibles et des features
TARGET_COLUMNS = ['Tmax_Demain', 'Tmin_Demain']
Y = df[TARGET_COLUMNS]
X = df.drop(columns=TARGET_COLUMNS)

# Séparation de l'ensemble de TEST
X_test = X[X.index >= TEST_SPLIT_DATE]
Y_test = Y[Y.index >= TEST_SPLIT_DATE]

# 2. Prédiction sur l'ensemble de Test
predictions = multi_output_model.predict(X_test)
predictions_df = pd.DataFrame(predictions, columns=['Tmax_Pred', 'Tmin_Pred'], index=Y_test.index)


# --- 3. VISUALISATION DE LA PERFORMANCE (Tmax) ---
plt.figure(figsize=(15, 6))
plt.plot(Y_test['Tmax_Demain'], label='Tmax Réel', color='blue', alpha=0.7)
plt.plot(predictions_df['Tmax_Pred'], label='Tmax Prédit (MAE: 1.70°C)', color='red', linestyle='--')
plt.title('Prédiction de la Température Maximale (Tmax) - Ensemble de Test (2018-2020)')
plt.xlabel('Date')
plt.ylabel('Température (°C)')
plt.legend()
plt.grid(True)
plt.show()



# --- 4. IMPORTANCE DES FEATURES (XGBoost) ---
# Nous utilisons l'importance des features du modèle qui prédit Tmax (premier estimateur)
importance = multi_output_model.estimators_[0].feature_importances_
feature_names = multi_output_model.estimators_[0].get_booster().feature_names

# Créer un DataFrame d'importance
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

print("\n--- IMPORTANCE DES FEATURES POUR LA PRÉDICTION Tmax ---")
print(feature_importance)

# Visualisation de l'importance des features
plt.figure(figsize=(12, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='skyblue')
plt.xlabel('Importance (F-Score)')
plt.title('Importance des Variables (Features) pour la Prévision Tmax J+1 (XGBoost)')
plt.gca().invert_yaxis() # La plus importante en haut
plt.show()