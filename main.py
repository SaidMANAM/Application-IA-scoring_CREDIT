import numpy as np
import pandas as pd
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from plotly.offline import iplot
import seaborn as sns
import zipfile

# Importing files from a zip repository and
path = r"\Users\Utilisateur\Downloads\Data_P7.zip"  #### le chemin vers le répertoire zip des données
with zipfile.ZipFile(path, "r") as zfile:
    dfs = {name[:-4]: pd.read_csv(zfile.open(name), encoding='cp1252')
           for name in zfile.namelist()
           }
    zfile.close()

def import_data():
    application_train = dfs['application_train']
    application_test = dfs['application_test']
    bureau = dfs['bureau']
    credit_card_balance = dfs['credit_card_balance']
    bureau_balance = dfs['bureau_balance']
    previous_application =  dfs['previous_application']

import_data()
# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(data, nan_as_category=True):
    original_columns = list(data.columns)
    categorical_columns = [col for col in data.columns if data[col].dtype == 'object']
    data = pd.get_dummies(data, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in data.columns if c not in original_columns]
    return data, new_columns


# table principale pour entrainement
def application_train_test(nan_as_category=False):
    # Read data and merge
    application_train = dfs['application_train']
    application_test = dfs['application_test']

    df = application_train.append(application_test).reset_index()
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df.drop(columns=['CODE_GENDER', 'NAME_TYPE_SUITE'])

    #   application_train = application_train[application_train['CODE_GENDER'] !='XNA']

    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])

    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    # Some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['CONSUMER_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, False)
    return df


# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(nan_as_category=True):
    bureau = dfs['bureau']
    bb = dfs['bureau_balance']
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)

    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
    del bb, bb_agg

    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat:
        cat_aggregations[cat] = ['mean']
    for cat in bb_cat:
        cat_aggregations[cat + "_MEAN"] = ['mean']

    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg

    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau

    return bureau_agg


# Preprocess previous_applications.csv
def previous_applications(nan_as_category=True):
    prev = dfs['previous_application']
    prev, cat_cols = one_hot_encoder(prev, nan_as_category=True)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']


    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev

    return prev_agg


# Preprocess POS_CASH_balance.csv
def pos_cash(nan_as_category=True):
    pos = dfs['POS_CASH_balance']
    pos, cat_cols = one_hot_encoder(pos, nan_as_category=True)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']

    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos

    return pos_agg


# Preprocess installments_payments.csv
def installments_payments(nan_as_category=True):
    ins = dfs['installments_payments']
    ins, cat_cols = one_hot_encoder(ins, nan_as_category=True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins

    return ins_agg


# Preprocess credit_card_balance.csv
def credit_card_balance(nan_as_category=True):
    cc = dfs['credit_card_balance']
    cc, cat_cols = one_hot_encoder(cc, nan_as_category=True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis=1, inplace=True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc

    return cc_agg


def distribution_plot(data, var, titre):
    plt.figure(figsize=(18, 8), dpi=100)
    plt.hist(data[var], density=True, stacked=True,   color="red")
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
    plt.pie(x=temp.values, labels=temp.index,  startangle=90,  autopct='%1.1f%%')
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
distribution_plot(application_train, 'AMT_CREDIT', "Distribution Amt_credit")
#   Distribution of the variable amount  annuity
distribution_plot(application_train, 'AMT_ANNUITY', "Distribution AMT_ANNUITY")
#   Distribution of the variable amount  goods  price
distribution_plot(application_train, 'AMT_GOODS_PRICE', "Distribution AMT_GOODS_PRICE")
#   Income Source of Applicant who applied for loan
bar_prc_plot(application_train, 'NAME_INCOME_TYPE', "Statut professionnel demandeurs de crédit", "Statut professionnel", "Pourcentage")
#   Occupation of Apllicant
bar_prc_plot(application_train, 'OCCUPATION_TYPE', "Fonction demandeurs de crédit", "Fonction", "Pourcentage")
#Education Of Applicant
bar_prc_plot(application_train, "NAME_EDUCATION_TYPE", "Niveau d'éducation demandeurs de crédit", "Education", "Pourcentage")
#   Family Status of Applicant
bar_prc_plot(application_train, 'NAME_FAMILY_STATUS', "Statut familial des demandeurs de crédit", "Statut familial", "Pourcentage")
#   housing type
bar_prc_plot(application_train, 'NAME_HOUSING_TYPE', "Type de logement des demandeurs de crédit", "Type de logement", "Pourcentage")
#   Loan repayed or not function of Income type  of  applicant
compare_plot(application_train, "NAME_INCOME_TYPE", 'Income source')
#   Loan repayed or not function of occupation type  of  applicant
compare_plot(application_train, "OCCUPATION_TYPE", 'Occupation')
#   Loan repayed or not function of organization type  of  applicant
compare_plot(application_train, "ORGANIZATION_TYPE", 'Organization')
#   Checking if data is unbalanced
bar_prc_plot(application_train, 'TARGET', 'la répartition des classes', 'classes', 'frequency')
#   Through which channel we acquired the client on the previous application
bar_prc_plot(previous_application, 'CHANNEL_TYPE', "Canal par lequel nous avons acquis le client sur l'application précédente", 'CHANNEL_TYPE', 'Frequency')
#   Status of previous  loans
pie_plot(previous_application, 'NAME_CONTRACT_STATUS', "Statut de crédits  demandés  avant")
#   Types of previous  loans
pie_plot(previous_application, "NAME_CONTRACT_TYPE", "Types de crédits  demandés  avant")
#   Types  of   loans
pie_plot(application_train, "NAME_CONTRACT_TYPE", "Types de crédits demandés")
#   Client Type of Previous Applications
pie_plot(previous_application, "NAME_CLIENT_TYPE", "Types de clients effectuant des  demandes précédantes")


def merging_data():
    df = application_train_test()
    bureau = bureau_and_balance()
    df = df.join(bureau, how='left', on='SK_ID_CURR')
    del bureau

    prev = previous_applications()
    df = df.join(prev, how='left', on='SK_ID_CURR')
    del prev

    pos = pos_cash()
    df = df.join(pos, how='left', on='SK_ID_CURR')
    del pos

    ins = installments_payments()
    df = df.join(ins, how='left', on='SK_ID_CURR')
    del ins

    cc = credit_card_balance()
    df = df.join(cc, how='left', on='SK_ID_CURR')
    del cc

    return df
