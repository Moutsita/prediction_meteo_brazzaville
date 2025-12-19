# Prévisions Météo Brazzaville et Analyse Climatique

Ce projet a été développé dans le cadre du **Master en Intelligence Artificielle** au **Dakar Institute of Technology (DIT)**. Il s'agit d'une solution de bout en bout permettant de prédire les températures extrêmes (Minimales et Maximales) pour la ville de Brazzaville à J+1 et J+2.

## Aperçu de l'Application
L'application fournit une interface interactive permettant de visualiser les observations réelles des 7 derniers jours et de générer des prévisions basées sur un modèle de Machine Learning entraîné sur des données historiques.

* **Interface :** Développée avec Streamlit.
* **Modèle :** Basé sur l'algorithme XGBoost.
* **Source de données :** Récupération en temps réel via l'API Meteostat.



## Architecture Technique
Le projet est structuré comme suit :
* `app.py` : Le script principal gérant l'interface utilisateur et la logique de prédiction.
* `requirements.txt` : Liste des dépendances Python nécessaires (XGBoost, Pandas, Streamlit, Meteostat).
* `models/` : Contient le modèle pré-entraîné exporté.
* `.streamlit/` : Fichiers de configuration pour le déploiement cloud.

## Performance du Modèle
Le modèle XGBoost a été validé avec les performances suivantes :
* **Erreur Absolue Moyenne (MAE) :** ~1.32 °C.
* **Horizon de prévision :** J+1 (Demain) et J+2 (Après-demain).

## Installation Locale
Pour exécuter ce projet sur votre machine :

1.  **Cloner le dépôt :**
    ```bash
    git clone [https://github.com/votre-utilisateur/votre-depot.git](https://github.com/votre-utilisateur/votre-depot.git)
    cd votre-depot
    ```

2.  **Installer les dépendances :**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Lancer l'application :**
    ```bash
    streamlit run app.py
    ```

## Déploiement
L'application est déployée sur **Streamlit Community Cloud** et est accessible via l'URL suivante : 
`https://votre-app.streamlit.app/`

## 👨‍🎓 Contexte Académique
* **Institution :** Dakar Institute of Technology (DIT)
* **Programme :** Master en Intelligence Artificielle
* **Auteur :** Moutsita
* **Année :** 2025