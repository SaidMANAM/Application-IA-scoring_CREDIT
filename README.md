# Modèle de scoring crédit

### Sommaire


- [Description](#description)
- [Compétences](#how-to-use)
- [Technologies](#references)

---

## Description

La mise en  oeuvre  d'un outil de “scoring crédit” pour calculer la probabilité qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. En développant un algorithme de classification en s’appuyant sur des sources de données variées. De plus un dashboard interactif a été  réalisé pour que les chargés de relation client puissent à la fois expliquer de façon la plus transparente possible les décisions d’octroi de crédit, mais également permettre à leurs clients de disposer de leurs informations personnelles et de les explorer facilement.

## Compétences

- import des données; Source des données : https://www.kaggle.com/c/home-credit-default-risk/data  
- Analyse exploratoire de données
- Feature engineering
- Supervised learning sur un jeu de données déséquilibré (pénalisation des classes avec SMOTE)
- Création d'une fonction cout métier adaptée à la problématique  
- Construction d'un modèle de scoring baseline
- Construction d'un modèle de scoring  avec Randomforest avec choix et  optimisation de ses hyperparamètres
- Construction d'un modèle de scoring  avec Lightgbm avec choix et  optimisation de ses hyperparamètres
- Ajout de la transparence  sur le travail du modèle avec SHAP
- Mise en place d'une API FastAPI pour appeler le modèle de prédiction
- Déploiement de l'API sur Heroku:https://api-scoring-fastapi.herokuapp.com  ( Repo utilisé: https://github.com/SaidMANAM/API_scoring_model_fastapi )
- Construction d'un dashboard interactif à destination des gestionnaires de relation client (Streamlit) et son déploiement

## Technologies

- FASTAPI
- SHAP
- STREAMLIT
- SCIKIT-LEARN
- Heroku
- SMOTE
- Python
- PANDAS
- GIT
- Pickle

[Back To The Top](#read-me-template)
