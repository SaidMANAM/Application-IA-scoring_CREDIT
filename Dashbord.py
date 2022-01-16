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
import sklearn
# from main import explain_model
import pickle
import shap

path_data = 'C:/Users/Utilisateur/OneDrive/Bureau/PROJET7/data_train.csv'
# df reduced : 10 % du jeu de donnees initial
path_valid = 'C:/Users/Utilisateur/OneDrive/Bureau/PROJET7/val.npy'
path_labels = 'C:/Users/Utilisateur/OneDrive/Bureau/PROJET7/labels.csv'
path_model = 'C:/Users/Utilisateur/OneDrive/Bureau/PROJET7/classifier.pkl'
path_validation = 'C:/Users/Utilisateur/OneDrive/Bureau/PROJET7/valid.csv'


@st.cache(allow_output_mutation=True)  # mise en cache de la fonction pour exécution unique
def chargement_data(path1, path2, path3, path4):
    dataframe = pd.read_csv(path1, dtype=np.float32)
    x_valid = np.load(path2)
    labels = pd.read_csv(path3, dtype=np.float32)
    validation = pd.read_csv(path4, dtype=np.float32)
    return dataframe, x_valid, labels, validation


@st.cache  # mise en cache de la fonction pour exécution unique
def chargement_model(path):
    with open(path, 'rb') as f:
        loaded_model = pickle.load(f)
    return loaded_model


def neighbor_model(x, y, id):
    if 'labels' in x.columns:
        x.drop(columns=['labels'], inplace=True)
    #x.set_index('SK_ID_CURR', inplace=True)
    pipeline_neighbors = sklearn.pipeline.Pipeline([('scaler', sklearn.preprocessing.StandardScaler()),
                                                    ('knn', sklearn.neighbors.KNeighborsClassifier(n_neighbors=5,
                                                                                                   algorithm='kd_tree'))])
    x_train, x_valid, y_train, y_valid = sklearn.model_selection.train_test_split(x, y, test_size=0.2, random_state=42)
    pipeline_neighbors.fit(x_train, y_train)
    nbrs = pipeline_neighbors['knn'].kneighbors(np.array(x_valid.loc[int(id)]).reshape(1, -1), return_distance=False)
    a = pd.DataFrame(x_valid.loc[int(id)]).transpose()[
        ['DAYS_EMPLOYED_PERC', 'AMT_ANNUITY', 'DAYS_BIRTH', 'AMT_GOODS_PRICE', 'AMT_INCOME_TOTAL', 'AMT_CREDIT']]
    a=a.append(x_train.iloc[list(nbrs[0])][
                ['DAYS_EMPLOYED_PERC', 'AMT_ANNUITY', 'DAYS_BIRTH', 'AMT_GOODS_PRICE', 'AMT_INCOME_TOTAL',
                 'AMT_CREDIT']])
    st.dataframe(a)


# @st.cache  # mise en cache de la fonction pour exécution unique
# def chargement_explanation(id, data, model, validation):
#     return explain_model(id, model, validation, data)

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


def explain_model(ide, model, data, X):
    if 'labels' in X.columns:
        X.drop(columns=['labels'], inplace=True)
    if X.shape[-1] == 770:
        X.set_index('SK_ID_CURR', inplace=True)
    explainer = shap.TreeExplainer(model, output_model="probability")
    shap_values = explainer.shap_values(data)
    sha_values = explainer(data)
    expected_value = explainer.expected_value
    idb = X.index.get_loc(float(ide))
    if isinstance(expected_value, list):
        expected_value = expected_value[1]
    select = range(2000)
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
    fig, ax = plt.subplots(nrows=1, ncols=1)
    shap.plots._waterfall.waterfall_legacy(expected_value, shap_values[idb], feature_names=list(X.columns),
                                           max_display=20)
    st.pyplot(fig)


st.set_page_config(page_title="Said's Dashboard",
                   page_icon="☮",
                   initial_sidebar_state="expanded")

dataframe, valid, labels, validation = chargement_data(path_data, path_valid, path_labels, path_validation)
liste_id = validation.index.tolist()
id_input = st.text_input('Veuillez saisir l\'identifiant du client:', )
dataframe['labels'] = labels.values

# requests.post('http://127.0.0.1:80/credit', data={'id': id_input})
# sample_en_regle = str(
#     list(dataframe[dataframe['labels'] == 0].sample(5)[['SK_ID_CURR', 'labels']]['SK_ID_CURR'].values)).replace('\'',
#                                                                                                                 '').replace(
#     '[', '').replace(']', '')
# chaine_en_regle = 'Exemples d\'id de clients en règle : ' + sample_en_regle
# sample_en_defaut = str(
#     list(dataframe[dataframe['labels'] == 1].sample(5)[['SK_ID_CURR', 'labels']]['SK_ID_CURR'].values)).replace('\'',
#                                                                                                                 '').replace(
#     '[', '').replace(']', '')
# chaine_en_defaut = 'Exemples d\'id de clients en défaut : ' + sample_en_defaut
model = chargement_model(path_model)

st.title('Dashbord  Scoring Credit Model')
st.subheader("Prédictions de scoring du client")

if id_input == '':  # rien n'a été saisi
    #st.write(chaine_en_defaut)
    #st.write(chaine_en_regle)
    st.write('Aucuni  n\'a été saisi')

#elif (int(id_input) in liste_id):  # quand un identifiant correct a été saisi on appelle l'API
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
        # classe_reelle = dataframe[dataframe['SK_ID_CURR'] == int(id_input)]['labels'].values[0]
        # classe_reelle = str(classe_reelle).replace('0', 'sans défaut').replace('1', 'avec défaut')
        # chaine = 'Prédiction : **' + etat + '** avec **' + str(
        #     round(proba * 100)) + '%** de risque de défaut (classe réelle :   ' + str(classe_reelle) + ')'

    #st.markdown(chaine)
    st.title("Explicabilité du modèle")

    explain_model(id_input, model['model'], valid, validation)
    st.title("Les clients ayant des caractéristiques des proches du demandeur:")
    neighbor_model(dataframe, labels, id_input)
    # # affichage de l'explication du score
    # with st.spinner('Chargement des détails de la prédiction...'):
    #     chargement_explanation(id_input, dataframe, model['model'], valid)
    columns1 = ['DAYS_EMPLOYED_PERC', 'AMT_ANNUITY', 'DAYS_BIRTH', 'AMT_GOODS_PRICE', 'AMT_INCOME_TOTAL', 'AMT_CREDIT']
    columns2 = ['DAYS_EMPLOYED_PERC', 'AMT_ANNUITY', 'DAYS_BIRTH', 'AMT_GOODS_PRICE', 'AMT_INCOME_TOTAL', 'AMT_CREDIT']
    columns3 = ['DAYS_EMPLOYED_PERC', 'AMT_ANNUITY', 'DAYS_BIRTH', 'AMT_GOODS_PRICE', 'AMT_INCOME_TOTAL', 'AMT_CREDIT']
    feature1 = st.selectbox('La liste des features', columns1)
    feature2 = st.selectbox('La liste des features2', columns2)
    feature3 = st.selectbox('La liste des features3', columns3)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    plt.scatter(validation[feature1],validation[feature2])
    plt.scatter(validation.loc[float(id_input)][feature1],validation.loc[float(id_input)][feature2], color="yellow")
    plt.title('Graphique scatter plot des variables: ' + feature1+' et '+feature2)
    st.pyplot(fig)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    plt.hist(validation[feature3])
    plt.axvline(x=validation.loc[float(id_input)][feature3],color="k")
    plt.title('Histogramme de la variable: '+feature3)
    st.pyplot(fig)


if __name__ == "__main__":

    print("Script runned directly")
else:
    print("Script called by other")
