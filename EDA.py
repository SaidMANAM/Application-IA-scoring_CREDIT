import numpy as np
import pandas as pd
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from plotly.offline import iplot
import seaborn as sns
import zipfile

path = r"\Users\Utilisateur\Downloads\Data_P7.zip"  #### le chemin vers le répertoire zip des données
with zipfile.ZipFile(path, "r") as zfile:
    dfs = {name[:-4]: pd.read_csv(zfile.open(name), encoding='cp1252')
           for name in zfile.namelist()
           }
    zfile.close()

def distribution_plot(data, var, titre):
    plt.figure(figsize=(18, 8), dpi=100)
    plt.hist(data[var], density=True, stacked=True, color="red")
    plt.title(titre)
    plt.show()


def bar_prc_plot(data, var, titre, xlab, ylab):
    fig = plt.figure(figsize=(14, 6), dpi=200)
    value = data[var].value_counts()
    value = (value / data[var].count() * 100)
    sns.barplot(y=value.values, x=value.index, hue=value.index, ci=100)
    plt.xlabel(xlab, size=14, color="red")
    plt.ylabel(ylab, size=14, color="red")
    plt.title(titre, size=15)
    plt.show()


def pie_plot(data, var, titre):
    plt.figure(figsize=(13, 5), dpi=200)
    temp = data[var].value_counts()
    plt.pie(x=temp.values, labels=temp.index, startangle=90, autopct='%1.1f%%')
    plt.title(titre)
    plt.show()


def compare_plot(data, var, title):
    temp = data[var].value_counts()
    # print(temp.values)
    temp_y0 = []
    temp_y1 = []
    for val in temp.index:
        temp_y1.append(np.sum(data["TARGET"][data[var] == val] == 1))
        temp_y0.append(np.sum(data["TARGET"][data[var] == val] == 0))
    trace1 = go.Bar(
        x=temp.index,
        y=(temp_y1 / temp.sum()) * 100,
        name='YES'
    )
    trace2 = go.Bar(
        x=temp.index,
        y=(temp_y0 / temp.sum()) * 100,
        name='NO'
    )

    data_ = [trace1, trace2]
    layout = go.Layout(
        title=title + " of Applicant's in terms of loan is repayed or not  in %",
        # barmode='stack',
        width=1000,
        xaxis=dict(
            title=title,
            tickfont=dict(
                size=14,
                color='rgb(107, 107, 107)'
            )
        ),
        yaxis=dict(
            title='Count in %',
            titlefont=dict(
                size=16,
                color='rgb(107, 107, 107)'
            ),
            tickfont=dict(
                size=14,
                color='rgb(107, 107, 107)'
            )
        )
    )

    fig = go.Figure(data=data_, layout=layout)
    iplot(fig)


application_train = dfs['application_train']
previous_application = dfs['previous_application']


#   Distribution of the variable amount  credit
distribution_plot(application_train, 'AMT_CREDIT', "Distribution de la  variable: Amt_credit")
#   Distribution of the variable amount  annuity
distribution_plot(application_train, 'AMT_ANNUITY', "Distribution de la  variable: AMT_ANNUITY")
#   Distribution of the variable amount  goods  price
distribution_plot(application_train, 'AMT_GOODS_PRICE', "Distribution de la  variable: AMT_GOODS_PRICE")
#   Income Source of Applicant who applied for loan
bar_prc_plot(application_train, 'NAME_INCOME_TYPE', "Statut professionnel demandeurs de crédit", "Statut professionnel",
             "Pourcentage")
#   Occupation of Apllicant
bar_prc_plot(application_train, 'OCCUPATION_TYPE', "Fonction demandeurs de crédit", "Fonction", "Pourcentage")
# Education Of Applicant
bar_prc_plot(application_train, "NAME_EDUCATION_TYPE", "Niveau d'éducation demandeurs de crédit", "Education",
             "Pourcentage")
#   Family Status of Applicant
bar_prc_plot(application_train, 'NAME_FAMILY_STATUS', "Statut familial des demandeurs de crédit", "Statut familial",
             "Pourcentage")
#   housing type
bar_prc_plot(application_train, 'NAME_HOUSING_TYPE', "Type de logement des demandeurs de crédit", "Type de logement",
             "Pourcentage")
#   Loan repayed or not function of Income type  of  applicant
compare_plot(application_train, "NAME_INCOME_TYPE", 'Income source')
#   Loan repayed or not function of occupation type  of  applicant
compare_plot(application_train, "OCCUPATION_TYPE", 'Occupation')
#   Loan repayed or not function of organization type  of  applicant
compare_plot(application_train, "ORGANIZATION_TYPE", 'Organization')
#   Checking if data is unbalanced
bar_prc_plot(application_train, 'TARGET', 'la répartition des classes', 'classes', 'frequency')
#   Through which channel we acquired the client on the previous application
bar_prc_plot(previous_application, 'CHANNEL_TYPE',
             "Canal par lequel nous avons acquis le client sur l'application précédente", 'CHANNEL_TYPE', 'Frequency')
#   Status of previous  loans
pie_plot(previous_application, 'NAME_CONTRACT_STATUS', "Statut de crédits  demandés  avant")
#   Types of previous  loans
pie_plot(previous_application, "NAME_CONTRACT_TYPE", "Types de crédits  demandés  avant")
#   Types  of   loans
pie_plot(application_train, "NAME_CONTRACT_TYPE", "Types de crédits demandés")
#   Client Type of Previous Applications
pie_plot(previous_application, "NAME_CLIENT_TYPE", "Types de clients effectuant des  demandes précédantes")