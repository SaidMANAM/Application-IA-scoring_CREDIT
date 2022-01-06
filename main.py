import numpy as np
import pandas as pd
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from plotly.offline import iplot
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn import pipeline
from collections import Counter
import pickle
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split, cross_val_predict, GridSearchCV, \
    KFold, RandomizedSearchCV, cross_validate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTETomek
import zipfile
import gc
import shap
import warnings

# from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import make_scorer

categorical_columns = []
# Importing files from a zip repository and
path = r"\Users\Utilisateur\Downloads\Data_P7.zip"  #### le chemin vers le répertoire zip des données
with zipfile.ZipFile(path, "r") as zfile:
    dfs = {name[:-4]: pd.read_csv(zfile.open(name), encoding='cp1252')
           for name in zfile.namelist()
           }
    zfile.close()


# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(data, nan_as_category=True, drop_first=True):
    original_columns = list(data.columns)
    cat_columns = [col for col in data.columns if data[col].dtype == 'object']
    data = pd.get_dummies(data, columns=cat_columns, dummy_na=nan_as_category, drop_first=drop_first)
    new_columns = [c for c in data.columns if c not in original_columns]
    return data, new_columns


# table principale pour entrainement
def application_train_test(nan_as_category=False, drop_first=True):
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
    df, cat_cols = one_hot_encoder(df, nan_as_category=nan_as_category, drop_first=drop_first)
    global categorical_columns
    categorical_columns = categorical_columns + list(df.select_dtypes(include='object').columns) + cat_cols
    return df


# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(nan_as_category=True, drop_first=True):
    bureau = dfs['bureau']
    bb = dfs['bureau_balance']
    bb, bb_cat = one_hot_encoder(bb, nan_as_category=nan_as_category, drop_first=drop_first)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category=nan_as_category, drop_first=False)
    global categorical_columns
    categorical_columns = categorical_columns + bureau_cat + bb_cat + list(
        bureau.select_dtypes(include='object').columns) + list(bb.select_dtypes(include='object').columns)
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
    del bb, bb_agg
    gc.collect()
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
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg


# Preprocess previous_applications.csv
def previous_applications(nan_as_category=True, drop_first=False):
    prev = dfs['previous_application']
    prev, cat_cols = one_hot_encoder(prev, nan_as_category=nan_as_category, drop_first=drop_first)
    global categorical_columns
    categorical_columns = categorical_columns + list(prev.select_dtypes(include='object').columns) + cat_cols
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
    gc.collect()
    return prev_agg


# Preprocess POS_CASH_balance.csv
def pos_cash(nan_as_category=True, drop_first=True):
    pos = dfs['POS_CASH_balance']
    pos, cat_cols = one_hot_encoder(pos, nan_as_category=nan_as_category, drop_first=drop_first)
    global categorical_columns
    categorical_columns = categorical_columns + list(pos.select_dtypes(include='object').columns) + cat_cols
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
    gc.collect()
    return pos_agg


# Preprocess installments_payments.csv
def installments_payments(nan_as_category=True, drop_first=True):
    ins = dfs['installments_payments']
    ins, cat_cols = one_hot_encoder(ins, nan_as_category=nan_as_category, drop_first=drop_first)
    global categorical_columns
    categorical_columns = categorical_columns + list(ins.select_dtypes(include='object').columns) + cat_cols
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
    gc.collect()
    return ins_agg


# Preprocess credit_card_balance.csv
def credit_card_balance(nan_as_category=True, drop_first=True):
    cc = dfs['credit_card_balance']
    cc, cat_cols = one_hot_encoder(cc, nan_as_category=nan_as_category, drop_first=drop_first)
    global categorical_columns
    categorical_columns = categorical_columns + list(cc.select_dtypes(include='object').columns) + cat_cols
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis=1, inplace=True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg


# def distribution_plot(data, var, titre):
#     plt.figure(figsize=(18, 8), dpi=100)
#     plt.hist(data[var], density=True, stacked=True, color="red")
#     plt.title(titre)
#     plt.show()
#
#
# def bar_prc_plot(data, var, titre, xlab, ylab):
#     fig = plt.figure(figsize=(14, 6), dpi=200)
#     value = data[var].value_counts()
#     value = (value / data[var].count() * 100)
#     sns.barplot(y=value.values, x=value.index, hue=value.index, ci=100)
#     plt.xlabel(xlab, size=14, color="red")
#     plt.ylabel(ylab, size=14, color="red")
#     plt.title(titre, size=15)
#     plt.show()
#
#
# def pie_plot(data, var, titre):
#     plt.figure(figsize=(13, 5), dpi=200)
#     temp = data[var].value_counts()
#     plt.pie(x=temp.values, labels=temp.index, startangle=90, autopct='%1.1f%%')
#     plt.title(titre)
#     plt.show()
#
#
# def compare_plot(data, var, title):
#     temp = data[var].value_counts()
#     # print(temp.values)
#     temp_y0 = []
#     temp_y1 = []
#     for val in temp.index:
#         temp_y1.append(np.sum(data["TARGET"][data[var] == val] == 1))
#         temp_y0.append(np.sum(data["TARGET"][data[var] == val] == 0))
#     trace1 = go.Bar(
#         x=temp.index,
#         y=(temp_y1 / temp.sum()) * 100,
#         name='YES'
#     )
#     trace2 = go.Bar(
#         x=temp.index,
#         y=(temp_y0 / temp.sum()) * 100,
#         name='NO'
#     )
#
#     data_ = [trace1, trace2]
#     layout = go.Layout(
#         title=title + " of Applicant's in terms of loan is repayed or not  in %",
#         # barmode='stack',
#         width=1000,
#         xaxis=dict(
#             title=title,
#             tickfont=dict(
#                 size=14,
#                 color='rgb(107, 107, 107)'
#             )
#         ),
#         yaxis=dict(
#             title='Count in %',
#             titlefont=dict(
#                 size=16,
#                 color='rgb(107, 107, 107)'
#             ),
#             tickfont=dict(
#                 size=14,
#                 color='rgb(107, 107, 107)'
#             )
#         )
#     )
#
#     fig = go.Figure(data=data_, layout=layout)
#     iplot(fig)
#
#
# application_train = dfs['application_train']
# previous_application = dfs['previous_application']


# #   Distribution of the variable amount  credit
# distribution_plot(application_train, 'AMT_CREDIT', "Distribution Amt_credit")
# #   Distribution of the variable amount  annuity
# distribution_plot(application_train, 'AMT_ANNUITY', "Distribution AMT_ANNUITY")
# #   Distribution of the variable amount  goods  price
# distribution_plot(application_train, 'AMT_GOODS_PRICE', "Distribution AMT_GOODS_PRICE")
# #   Income Source of Applicant who applied for loan
# bar_prc_plot(application_train, 'NAME_INCOME_TYPE', "Statut professionnel demandeurs de crédit", "Statut professionnel",
#              "Pourcentage")
# #   Occupation of Apllicant
# bar_prc_plot(application_train, 'OCCUPATION_TYPE', "Fonction demandeurs de crédit", "Fonction", "Pourcentage")
# # Education Of Applicant
# bar_prc_plot(application_train, "NAME_EDUCATION_TYPE", "Niveau d'éducation demandeurs de crédit", "Education",
#              "Pourcentage")
# #   Family Status of Applicant
# bar_prc_plot(application_train, 'NAME_FAMILY_STATUS', "Statut familial des demandeurs de crédit", "Statut familial",
#              "Pourcentage")
# #   housing type
# bar_prc_plot(application_train, 'NAME_HOUSING_TYPE', "Type de logement des demandeurs de crédit", "Type de logement",
#              "Pourcentage")
# #   Loan repayed or not function of Income type  of  applicant
# compare_plot(application_train, "NAME_INCOME_TYPE", 'Income source')
# #   Loan repayed or not function of occupation type  of  applicant
# compare_plot(application_train, "OCCUPATION_TYPE", 'Occupation')
# #   Loan repayed or not function of organization type  of  applicant
# compare_plot(application_train, "ORGANIZATION_TYPE", 'Organization')
# #   Checking if data is unbalanced
# bar_prc_plot(application_train, 'TARGET', 'la répartition des classes', 'classes', 'frequency')
# #   Through which channel we acquired the client on the previous application
# bar_prc_plot(previous_application, 'CHANNEL_TYPE',
#              "Canal par lequel nous avons acquis le client sur l'application précédente", 'CHANNEL_TYPE', 'Frequency')
# #   Status of previous  loans
# pie_plot(previous_application, 'NAME_CONTRACT_STATUS', "Statut de crédits  demandés  avant")
# #   Types of previous  loans
# pie_plot(previous_application, "NAME_CONTRACT_TYPE", "Types de crédits  demandés  avant")
# #   Types  of   loans
# pie_plot(application_train, "NAME_CONTRACT_TYPE", "Types de crédits demandés")
# #   Client Type of Previous Applications
# pie_plot(previous_application, "NAME_CLIENT_TYPE", "Types de clients effectuant des  demandes précédantes")


def merging_data(random_state):
    data = application_train_test()
    bureau = bureau_and_balance()
    data = data.join(bureau, how='left', on='SK_ID_CURR')
    del bureau

    prev = previous_applications()
    data = data.join(prev, how='left', on='SK_ID_CURR')
    del prev

    pos = pos_cash()
    data = data.join(pos, how='left', on='SK_ID_CURR')
    del pos

    ins = installments_payments()
    data = data.join(ins, how='left', on='SK_ID_CURR')
    del ins

    cc = credit_card_balance()
    data = data.join(cc, how='left', on='SK_ID_CURR')
    del cc
    gc.collect()
    global categorical_columns
    b = list(set(categorical_columns) - (set(categorical_columns) - set(list(data.columns))))
    data = data.dropna(subset=['TARGET'])
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data[b] = data[b].fillna(-1)
    a = list(set(data.columns) - set(b))
    data[a] = data[a].fillna(data[a].median())
    data = data.dropna()
    data = reduce_mem_usage(data)
    y = data['TARGET']
    # data = data.drop(['TARGET'], axis=1)
    data.set_index('SK_ID_CURR', inplace=True)
    data = data.drop(columns=['TARGET', 'index'], axis=1)

    data.info(memory_usage='deep')
    columns = list(data.columns)
    x_tr, x_val, y_tr, y_val = train_test_split(data, y, test_size=0.2, random_state=random_state)
    scaler = StandardScaler()
    scaler.fit(x_tr)
    x_tr = scaler.transform(x_tr)
    x_vali = scaler.transform(x_val)
    x_tr =x_tr.astype(np.float32)
    y_val =y_val.astype(np.float32)
    y_tr =y_tr.astype(np.float32)
    x_vali = x_vali.astype(np.float32)
    print(type(x_tr))
    print(type(x_val))
    print(type(x_vali))
    data.to_csv(path_or_buf=r'C:/Users/Utilisateur/OneDrive/Bureau/PROJET7/data_train.csv')
    np.save(r'C:/Users/Utilisateur/OneDrive/Bureau/PROJET7/val', x_vali)
    y.to_csv(r'C:/Users/Utilisateur/OneDrive/Bureau/PROJET7/labels.csv', index=False)
    return x_tr, x_vali, y_tr, y_val, columns, scaler, x_val, data, y


def mem_usage(pandas_obj):
    if isinstance(pandas_obj, pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else:
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2  # convertir les bytes en megabytes
    return "{:03.2f} MB".format(usage_mb)  # afficher sous format nombre (min 3 chiffres) et une précisionµ


def cost_function(y_true, y_pred, **kwargs):
    # pred = estimator.predict(X)
    global x
    cost = (((y_pred == 0) & (y_true == 0)) * x['AMT_CREDIT'] * (0.03)
            - ((y_pred == 1) & (y_true == 0)) * x['AMT_CREDIT'] * (0.03)
            - ((y_pred == 0) & (y_true == 1)) * x['AMT_CREDIT'] * (1 + 0.03))
    return np.sum(cost)


def reduce_mem_usage(df1):
    df1.info(memory_usage='deep')
    df_float = df1.select_dtypes(include=['float']).copy()
    converted_float = df_float.apply(pd.to_numeric, downcast='float')
    df_int = df1.select_dtypes(include=['int']).copy()
    converted_int = df_int.apply(pd.to_numeric, downcast='integer')
    # converted_float.info(memory_usage='deep')
    a = mem_usage(df_int)
    b = mem_usage(converted_int)
    a1 = mem_usage(df_float)
    b1 = mem_usage(converted_float)
    a2 = (float(a1.replace('MB', '')) + float(a.replace('MB', '')))
    b2 = (float(b1.replace('MB', '')) + float(b.replace('MB', '')))
    c = 100 * (float(b1.replace('MB', '')) + float(b.replace('MB', ''))) / (
            float(a1.replace('MB', '')) + float(a.replace('MB', '')))
    print("L'utilisation de la mémoire avant traitement:{}".format(a2))
    print("L'utilisation de la mémoire après traitement:{}".format(b2))
    print("le gain en mémoire est de:{:.2f}%".format(c))
    df1[converted_float.columns] = converted_float
    df1[converted_int.columns] = converted_int
    del a, b, a1, a2, b1, b2, c
    gc.collect()
    return df1


x_train, x_valid, y_train, y_valid, list_colonnes, std_scaler, x_exp, x, Y = merging_data(42)


def data_balance(data, y):
    count = Counter(y)
    percent = {key: 100 * value / len(y) for key, value in count.items()}
    print("avant de balancer le dataset, la répartiti   ondes   classes en  %   était:", percent)  # pourcentage
    # data = reduce_mem_usage(data)
    balancer = SMOTETomek(random_state=42)
    data, y = balancer.fit_resample(data, y)
    count = Counter(y)
    percent = {key: 100 * value / len(y) for key, value in count.items()}
    print("Les classes après  balance dataset:", percent)
    return data, y


def performance(y, prediction):
    print("accuracy", accuracy_score(y, prediction))
    print("f1 score macro", f1_score(y, prediction, average='micro'))
    print("precision score", precision_score(y, prediction, average='micro'))
    print("recall score", recall_score(y, prediction, average='micro'))
    print("classification_report    \n", classification_report(y, prediction))


# def random_classifier(random_state, x_tr, x_val, y_tr, y_val):
#     dummy_clf = DummyClassifier(strategy="stratified", random_state=random_state).fit(x_tr, y_tr)
#     predicted = cross_val_predict(dummy_clf, x_tr, y_tr)
#     print("Performances en phase d'entrainement")
#     performance(y_tr, predicted)
#     predicted_valid = cross_val_predict(dummy_clf, x_val, y_val)
#     print("Performances en phase de test")
#     performance(y_val, predicted_valid)
#     print("Training AUC&ROC", roc_auc_score(y_tr, dummy_clf.predict_proba(x_tr)[:, 1]))
#     print("Testing AUC&ROC", roc_auc_score(y_val,  dummy_clf.predict_proba(x_val)[:, 1]))
#     return dummy_clf
#
# random_classifier(42,x_train, x_valid, y_train, y_valid)
#
#
# def random_forest_classifier(random_state, x_tr, x_val, y_tr, y_val):
#     over = SMOTE(random_state=42,sampling_strategy=0.1)
#     under = RandomUnderSampler(sampling_strategy=0.5)
#     steps = [('over', over),('under', under), ('model', RandomForestClassifier(random_state=random_state))]
#     pipe = Pipeline(steps=steps)
#     cv = KFold(n_splits=3)
#
#     # Number of trees in random forest
#     n_estimators = np.linspace(start=10, stop=80, num=10, dtype=int)
#      # Number of features to consider at every split
#     max_features = ['auto', 'sqrt']
#
#      # Maximum number of levels in tree
#     max_depth = [2, 4]
# #     # Minimum number of samples required to split a node
#     min_samples_split = [2, 5]
# #     # Minimum number of samples required at each leaf node
#     min_samples_leaf = [1, 2]
# #     # Method of selecting samples for training each tree
#     bootstrap = [True, False]
# #
# #     # Create the param grid
#     param_grid = {'model__n_estimators': n_estimators,
#                   'model__max_features': max_features,
#                   'model__max_depth': max_depth,
#                   'model__min_samples_split': min_samples_split,
#                   'model__min_samples_leaf': min_samples_leaf,
#                   'model__bootstrap': bootstrap
#                   }
#     score = make_scorer(cost_function, greater_is_better=True)
#     grid_cv = RandomizedSearchCV(estimator=pipe, param_distributions=param_grid, n_iter=5, cv=cv, scoring='roc_auc',
#                                  random_state=random_state, n_jobs=-1,  verbose=True, refit=True,error_score='raise')
#     print('cross validation')
#     grid_cv.fit(x_tr, y_tr)
#     best_params = grid_cv.best_params_
#     best_model = grid_cv.best_estimator_
#     scores= cross_validate( best_model, x_tr, y=y_tr, scoring='roc_auc', cv=5, verbose=True,  return_train_score=True, return_estimator=True)
#     print('Train Area Under the Receiver Operating Characteristic Curve - : {:.3f} +/- {:.3f}'.format(scores['train_score'].mean(),scores['train_score'].std()))
#     print('Validation Area Under the Receiver Operating Characteristic Curve - : {:.3f} +/- {:.3f}'.format(scores['test_score'].mean(),scores['test_score'].std()))
#     print('Test Area Under the Receiver Operating Characteristic Curve - : {:.3f}'.format(roc_auc_score(y_val, best_model['model'].predict_proba(x_val)[:, 1])))
#     print(best_model)
#     return best_params, best_model

# params, rf_model = random_forest_classifier(42, x_train, x_valid, y_train, y_valid)


def lightgbm_classifier(random_state, x_tr, x_val, y_tr, y_val):
    import lightgbm as lgb
    from scipy.stats import randint as sp_randint
    from scipy.stats import uniform as sp_uniform
    over = SMOTE(random_state=42, sampling_strategy=0.1)
    under = RandomUnderSampler(sampling_strategy=0.5)
    cv = KFold(n_splits=3)
    balancer = SMOTE(random_state=random_state)
    steps = [('over', over), ('under', under), ('model', lgb.LGBMClassifier())]
    pipe = Pipeline(steps=steps)
    param_test = {'model__num_leaves': sp_randint(14, 50),
                  'model__max_depth': sp_randint(4, 10),
                  'model__min_child_samples': sp_randint(100, 500),
                  'model__min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
                  'model__subsample': sp_uniform(loc=0.2, scale=0.8),
                  'model__colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
                  'model__reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
                  'model__reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}
    score = make_scorer(cost_function, greater_is_better=True)
    n_points_to_test = 100
    gs = RandomizedSearchCV(
        estimator=pipe, param_distributions=param_test,
        n_iter=2,
        scoring='roc_auc',
        cv=cv,
        refit=True,
        random_state=random_state,
        verbose=True,
        n_jobs=-1)
    print('cross validation')
    gs.fit(x_tr, y_tr)
    best_params = gs.best_params_
    best_model = gs.best_estimator_
    scores = cross_validate(best_model, x_tr, y=y_tr, scoring='roc_auc', cv=5, verbose=True, return_train_score=True,
                            return_estimator=True)
    print('Train Area Under the Receiver Operating Characteristic Curve - : {:.3f} +/- {:.3f}'.format(
        scores['train_score'].mean(), scores['train_score'].std()))
    print('Validation Area Under the Receiver Operating Characteristic Curve - : {:.3f} +/- {:.3f}'.format(
        scores['test_score'].mean(), scores['test_score'].std()))
    print('Test Area Under the Receiver Operating Characteristic Curve - : {:.3f}'.format(
        roc_auc_score(y_val, best_model['model'].predict_proba(x_val)[:, 1])))
    print(best_params)
    return best_params, best_model


def explain_model(ide,model,data, X):
    explainer = shap.TreeExplainer(model, output_model="probability")
    shap_values = explainer.shap_values(data)
    sha_values = explainer(data)
    expected_value = explainer.expected_value
    idb= X.index.get_loc(ide)
    if isinstance(expected_value, list):
        expected_value = expected_value[1]
    select = range(2000)
    features_display = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values = explainer.shap_values(features_display)[1]
    shap.summary_plot(shap_values,data,feature_names=list(X.columns), title='Graphe des variables les plus influantes sur la décision du modèle ')
    shap.decision_plot(expected_value, shap_values[idb],data[idb], feature_names=list(X.columns), ignore_warnings=True,title='Graphe d\'explication de la décision du modèle ')
    shap.initjs()
    shap.force_plot(expected_value, shap_values[idb],feature_names=list(X.columns))
    shap.plots._waterfall.waterfall_legacy(expected_value, shap_values[idb], feature_names=list(X.columns), max_display=20)


def pipeline_trained(model, params, scaler):
    pipeline_pred = pipeline.Pipeline([('scaler', scaler),
                                       ('model', model(**params))])
    pipeline_pred.fit(X, Y)
    pickle_out = open(r'classifier.pkl', "wb")
    pickle.dump(pipeline_pred, pickle_out)
    pickle_out.close()
    return "model entrainé et serialisé"


best_params, best_model = lightgbm_classifier(42, x_train, x_valid, y_train, y_valid)
#explain_model(10004,best_model["model"], x_valid, x)
