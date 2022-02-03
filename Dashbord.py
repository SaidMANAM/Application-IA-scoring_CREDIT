import pandas as pd
import numpy as np
import streamlit as st
import requests
import json
from urllib.request import urlopen
import ast
import warnings
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import os
import pickle
import shap


path_data = r'C:\Users\Utilisateur\OneDrive\Bureau\PROJET7\data_tain.pkl'
# path_data = os.getcwd() +'/data_train.csv'
# df reduced : 10 % du jeu de donnees initial
path_valid = os.getcwd() + '/val.npy'
path_labels = os.getcwd() + '/y_valid.csv'
path_model = os.getcwd() + '/classifier.pkl'
path_validation = os.getcwd() + '/x_valid.csv'
#path = "https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/Projet+-+Impl%C3%A9menter+un+mod%C3%A8le+de+scoring/Projet+Mise+en+prod+-+home-credit-default-risk.zip"


# @st.cache(allow_output_mutation=True)  # mise en cache de la fonction pour exécution unique
# def chargement_data(path1):
#     url = urllib.request.urlopen(path1)
#     with zipfile.ZipFile(BytesIO(url.read())) as zfile:
#         dfs = {name[:-4]: pd.read_csv(zfile.open(name), encoding='cp1252')
#                for name in zfile.namelist()
#                }
#         zfile.close()
#     data, y = merging_data()
# @st.cache(allow_output_mutation=True)  # mise en cache de la fonction pour exécution unique
# def chargement_data(random_state):
#     return  prepare_data(random_state)
@st.cache(allow_output_mutation=True)  # mise en cache de la fonction pour exécution unique
#def chargement_data(path1, path2, path3, path4):
def chargement_data(  path3, path4):

    # dataframe = pd.read_csv(path1, dtype=np.float32)
    # with open(path1, 'rb') as f:
    #     dataframe = pickle.load(f)
    # if dataframe.shape[-1] == 770:
    #     dataframe.set_index('SK_ID_CURR', inplace=True)
    #valid_np = np.load(path2)
    labels = pd.read_csv(path3, dtype=np.float32)
    x_validation = pd.read_csv(path4, dtype=np.float32)
    if x_validation.shape[-1] == 770:
        x_validation.set_index('SK_ID_CURR', inplace=True)
    #return dataframe, valid_np, labels, x_validation
    return labels, x_validation


@st.cache  # mise en cache de la fonction pour exécution unique
def chargement_model(path):
    with open(path, 'rb') as f:
        loaded_model = pickle.load(f)
    return loaded_model


# def neighbor_model(x, y, id):
#     if 'labels' in x.columns:
#         x.drop(columns=['labels'], inplace=True)
#     if x.shape[-1] == 770:
#         x.set_index('SK_ID_CURR', inplace=True)
#     #x.set_index('SK_ID_CURR', inplace=True)
#     pipeline_neighbors = sklearn.pipeline.Pipeline([('scaler', sklearn.preprocessing.StandardScaler()),
#                                                     ('knn', sklearn.neighbors.KNeighborsClassifier(n_neighbors=5,
#                                                                                                    algorithm='kd_tree'))])
#     x_train, x_valid, y_train, y_valid = sklearn.model_selection.train_test_split(x, y, test_size=0.2, random_state=42)
#     pipeline_neighbors.fit(x_train, y_train)
#     nbrs = pipeline_neighbors['knn'].kneighbors(np.array(x_valid.loc[int(id)]).reshape(1, -1), return_distance=False)
#     a = pd.DataFrame(x_valid.loc[int(id)]).transpose()[
#         ['DAYS_EMPLOYED_PERC', 'AMT_ANNUITY', 'DAYS_BIRTH', 'AMT_GOODS_PRICE', 'AMT_INCOME_TOTAL', 'AMT_CREDIT']]
#     a=a.append(x_train.iloc[list(nbrs[0])][
#                 ['DAYS_EMPLOYED_PERC', 'AMT_ANNUITY', 'DAYS_BIRTH', 'AMT_GOODS_PRICE', 'AMT_INCOME_TOTAL',
#                  'AMT_CREDIT']])
#     st.dataframe(a)


def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


def explain_model(ide, model, data, X):
    if 'labels' in X.columns:
        X.drop(columns=['labels'], inplace=True)
    if X.shape[-1] == 770:
        X.set_index('SK_ID_CURR', inplace=True)
    explainer = shap.TreeExplainer(model, output_model="probability")
    expected_value = explainer.expected_value
    idb = X.index.get_loc(float(ide))
    if isinstance(expected_value, list):
        expected_value = expected_value[1]
    features_display = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values = explainer.shap_values(features_display)[1]
    st.header('Explicabilité Globale ')
    st.subheader('Summary Plot')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    shap.summary_plot(shap_values, data, feature_names=list(X.columns),
                      title='Graphe des variables les plus influantes sur la décision du modèle ')
    st.pyplot(fig)
    st.header('Explicabilité Locale ')
    st.subheader('Force Plot')
    shap.initjs()
    # shap.force_plot(expected_value, shap_values[idb], feature_names=list(X.columns))

    st_shap((shap.force_plot(expected_value, shap_values[idb], feature_names=list(X.columns))), 200)
    # st.pyplot(fig)

    st.subheader('Decision Plot')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    shap.decision_plot(expected_value, shap_values[idb], data[idb], feature_names=list(X.columns),
                       ignore_warnings=True, title='Graphe d\'explication de la décision du modèle ')
    st.pyplot(fig)
    st.subheader('Waterfall Plot')


st.set_page_config(page_title="Said's Dashboard",
                   page_icon="☮",
                   initial_sidebar_state="expanded")

#dataframe, valid, labels, validation = chargement_data(path_data, path_valid, path_labels, path_validation)

labels, x_validation = chargement_data(path_labels, path_validation)
model = chargement_model(path_model)
if x_validation.shape[-1] == 770:
    x_validation.set_index('SK_ID_CURR', inplace=True)
valid_np=model['scaler'].transform(x_validation)
liste_id = x_validation.index.tolist()
id_input = st.text_input('Veuillez saisir l\'identifiant du client:', )
x_validation['labels'] = labels.values

# requests.post('http://127.0.0.1:80/credit', data={'id': id_input})
# sample_en_regle = str(
#     list(dataframe[dataframe['labels'] == 0].sample(5)[['SK_ID_CURR', 'labels']]['SK_ID_CURR'].values)).replace('\'',
#                                                                                                                 '').replace(
#     '[', '').replace(']', '')
# sample_en_regle=str(list((labels[246008:]==0).sample(5).index)).replace('[', '').replace(']', '')
# chaine_en_regle = 'Exemples d\'id de clients en règle : ' + sample_en_regle
# sample_en_defaut = str(
#     list(dataframe[dataframe['labels'] == 1].sample(5)[['SK_ID_CURR', 'labels']]['SK_ID_CURR'].values)).replace('\'',
#                                                                                                                 '').replace(
#     '[', '').replace(']', '')
# chaine_en_defaut = 'Exemples d\'id de clients en défaut : ' + sample_en_defaut


st.title('Dashbord  Scoring Credit Model')
st.subheader("Prédictions de scoring du client")

if id_input == '':  # rien n'a été saisi
    # st.write(chaine_en_defaut)
    # st.write(chaine_en_regle)
    st.write('Aucun ID  n\'a été saisi')

# elif (int(id_input) in liste_id):  # quand un identifiant correct a été saisi on appelle l'API
elif (float(id_input) in liste_id):
    # Appel de l'API :

    API_url = "http://127.0.0.1:80/credit/" + str(id_input)
    with st.spinner('Chargement du score du client...'):
        print(API_url)
        json_url = urlopen(API_url)
        print(json_url)
        API_data = json.loads(json_url.read())
        print(type(ast.literal_eval(API_data)))
        results = ast.literal_eval(API_data)
        classe_predite = results['prediction']
        print(classe_predite)
        if classe_predite == 1:
            etat = 'client à risque'
        else:
            etat = 'client peu risqué'
        proba = 1 - results['proba_remboureser']
        prediction = results['prediction']
        classe_reelle = x_validation.loc[int(id_input)]['labels']
        classe_reelle = str(classe_reelle).replace('0', 'sans défaut').replace('1', 'avec défaut')
        chaine = 'Prédiction : **' + etat + '** avec **' + str(
            round(proba * 100)) + '%** de rembourser (classe réelle :   ' + str(classe_reelle) + ')'

    st.markdown(chaine)
    st.title("Explicabilité du modèle")

    explain_model(id_input, model['model'], valid_np, x_validation)

    st.title("Les clients ayant des caractéristiques des proches du demandeur:")
    # neighbor_model(dataframe, labels, id_input)
    # # affichage de l'explication du score
    # with st.spinner('Chargement des détails de la prédiction...'):
    # chargement_explanation(id_input, dataframe, model['model'], valid)
    threshold = results['treshold']
    fig, ax = plt.subplots()
    if proba < threshold:
        color = 'r'
    else:
        color = 'g'
    ax.bar('prediction', height=proba, width=0.5, color=color)
    plt.title('Niveau de la confiance du remboursement de pret')
    # ax.bar('prediction', height=np.minimum(threshold, proba), width=0.5, color="b")
    # ax.bar('prediction', height=abs(proba - threshold), width=0.5, color="pink",
    # bottom=np.minimum(threshold, proba))
    plt.axhline(y=threshold, linewidth=0.5, color='k')
    plt.ylim([0, 1])
    ax.text(0, threshold + 0.1, str(threshold))
    st.pyplot(fig)

    columns1 = ['', 'EXT_SOURCE_3', 'EXT_SOURCE_2', 'EXT_SOURCE_1', 'PAYMENT_RATE', 'DAYS_EMPLOYED',
                'CONSUMER_GOODS_RATIO',
                'AMT_ANNUITY', 'DAYS_BIRTH', 'AMT_GOODS_PRICE', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'ANNUITY_INCOME_PERC']
    feature1 = st.selectbox('La liste des features', columns1)
    feature2 = st.selectbox('La liste des features2', columns1)

    if feature1 != '':
        fig, ax = plt.subplots(nrows=1, ncols=1)
        plt.hist(x_validation[feature1])
        plt.axvline(x=x_validation.loc[float(id_input)][feature1], color="k")
        plt.title('Histogramme de la variable: ' + feature1)
        st.pyplot(fig)
    if feature2 != '':
        fig, ax = plt.subplots(nrows=1, ncols=1)
        plt.hist(x_validation[feature2])
        plt.axvline(x=x_validation.loc[float(id_input)][feature2], color="k")
        plt.title('Histogramme de la variable: ' + feature2)
        st.pyplot(fig)
    if feature2 != '' and feature1 != '':
        fig, ax = plt.subplots(nrows=1, ncols=1)
        plt.scatter(x_validation[feature1], x_validation[feature2])
        plt.scatter(x_validation.loc[float(id_input)][feature1], x_validation.loc[float(id_input)][feature2],
                    color="yellow")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.title('Graphique scatter plot des variables: ' + feature1 + ' et ' + feature2)
        st.pyplot(fig)

if __name__ == "__main__":

    print("Script runned directly")
else:
    print("Script called by other")
