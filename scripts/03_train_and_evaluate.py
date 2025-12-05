import pandas as pd
import numpy as np
import os
import sys
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor

# --- CONFIGURATION ---
INPUT_PATH = 'data/features_finales.csv'
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'final_model.pkl')

VALIDATION_SPLIT_DATE = '2016-01-01'
TEST_SPLIT_DATE = '2018-01-01'
# --- FIN CONFIGURATION ---

# 1. Assurer que le dossier 'models' existe
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# 2. Chargement des données
try:
    df = pd.read_csv(INPUT_PATH, index_col='time', parse_dates=True)
except FileNotFoundError:
    print(f"Erreur: Le fichier de features {INPUT_PATH} est introuvable. Exécutez le script 02 en premier.")
    sys.exit(1)

# 1. Définir les colonnes cibles (les deux qui se trouvent à la fin de features_finales.csv)
TARGET_COLUMNS = ['Tmax_Demain', 'Tmin_Demain'] 

# 2. Y est le DataFrame contenant les deux cibles (Shape: (n_samples, 2))
Y = df[TARGET_COLUMNS]

# 3. X est le DataFrame contenant toutes les autres colonnes (les features)
X = df.drop(columns=TARGET_COLUMNS)
# --- FIN DE LA CORRECTION ---

"""
X = df.drop(columns=['Tmax_Demain'])
Y = df['Tmax_Demain']
"""

# 3. SÉPARATION CHRONOLOGIQUE TRAIN/VALIDATION/TEST
X_train = X[X.index < VALIDATION_SPLIT_DATE]
Y_train = Y[Y.index < VALIDATION_SPLIT_DATE]

X_val = X[(X.index >= VALIDATION_SPLIT_DATE) & (X.index < TEST_SPLIT_DATE)]
Y_val = Y[(Y.index >= VALIDATION_SPLIT_DATE) & (Y.index < TEST_SPLIT_DATE)]

X_test = X[X.index >= TEST_SPLIT_DATE]
Y_test = Y[Y.index >= TEST_SPLIT_DATE]

print(f"Train: {len(X_train)} jours (1991 - 2015)")
print(f"Validation: {len(X_val)} jours (2016 - 2017)")
print(f"Test Final: {len(X_test)} jours (2018 - 2020)")


# 4. ENTRAÎNEMENT DU MODÈLE XGBOOST MULTI-SORTIE
print("\nDébut de l'entraînement du modèle XGBoost Multi-Sortie...")

# 1. Définition du régresseur de base
base_model = XGBRegressor(
    n_estimators=5000, 
    learning_rate=0.01,
    max_depth=5,
    n_jobs=-1,
    random_state=42
)

# 2. Utilisation du wrapper MultiOutputRegressor
multi_output_model = MultiOutputRegressor(base_model)

# Pour la validation dans MultiOutputRegressor, nous utilisons un fit simple 
# car le wrapper ne supporte pas nativement early_stopping_rounds comme before.
# Pour simplifier, nous utilisons ici le fit standard.

multi_output_model.fit(X_train, Y_train) 

# 5. ÉVALUATION FINALE (sur l'ensemble de TEST)
predictions = multi_output_model.predict(X_test)
predictions_df = pd.DataFrame(predictions, columns=['Tmax_Pred', 'Tmin_Pred'], index=Y_test.index)

# Calcul des métriques pour chaque sortie
mae_max = mean_absolute_error(Y_test['Tmax_Demain'], predictions_df['Tmax_Pred'])
mae_min = mean_absolute_error(Y_test['Tmin_Demain'], predictions_df['Tmin_Pred'])

print(f"\n--- RÉSULTATS D'ÉVALUATION FINALE (2018-2020) ---")
print(f"Erreur Absolue Moyenne Tmax : {mae_max:.2f} °C")
print(f"Erreur Absolue Moyenne Tmin : {mae_min:.2f} °C")
print(f"Erreur Globale Moyenne (MAE) : {np.mean([mae_max, mae_min]):.2f} °C")
print("-----------------------------------------------------")

# 6. SAUVEGARDE DU MODÈLE
joblib.dump(multi_output_model, MODEL_PATH)
print(f"Modèle Multi-Sortie sauvegardé sous : {MODEL_PATH}")