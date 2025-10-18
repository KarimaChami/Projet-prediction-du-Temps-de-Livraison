# Projet de Prédiction du Temps de Livraison
## Objectif du projet
Ce projet a pour but de **prédire le temps de livraison (`Delivery_Time_min`)** à partir de différentes caractéristiques liées aux commandes (distance, conditions météo, expérience du livreur, etc.).  
Il s'agit d'un projet de **data science** appliqué, combinant :
- Nettoyage des données
- Prétraitement et encodage
- Sélection de features
- Modélisation avec **GridSearchCV**
- Évaluation des performances
- Création d’un **pipeline automatisé**
- Tests automatisés avec **pytest**

---

## Structure du projet
├── data/
      -dataset.csv
├── delivery_model.py #Préparation, encodage, split, entraînement des modèles
├── EDA.ipynb  # Exploration et visualisation des données
├── requirements.txt
├── test_pipeline.py # Tests unitaires pour valider la cohérence du pipeline

## Test
Le fichier test_pipeline.py exécuté avec Pytest vérifie :

La cohérence des dimensions entre X et y
=> un test unitaire validé la cohérence des dimensions des données après séparation entre X et y.
=> un test unitaire Vérification que la MAE maximale ne dépasse pas un seuil défini

##  Modèles entraînés
Deux modèles ont été comparés :
- **SVR**
- **Random Forest Regressor**


## Modèle retenu pour la mise en production
Le modèle choisi est la **SVR** car il offre :  
- environ 80% de la variance des temps de livraison est expliquée par les features sélectionnées.
- Bonne généralisation 
- Meilleure précision moyenne après optimisation des hyperparamètres 

## Intégration Continue (CI)
Dans le cadre de ce projet, j’ai mis en place un système d’intégration continue (CI) avec GitHub Actions pour :
 - Exécuter automatiquement les tests unitaires à chaque push
 - Vérifier la cohérence du pipeline et la performance du modèle
 - Garantir la stabilité du projet avant tout déploiement

## Exécution du projet
### Installation des dépendances
    ```bash
    pip install -r requirements.txt

