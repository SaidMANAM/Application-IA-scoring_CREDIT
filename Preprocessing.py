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
from io import BytesIO
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTETomek
import zipfile
import gc
import shap
import warnings
import urllib
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import make_scorer


#Importing files 
path= "https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/Projet+-+Impl%C3%A9menter+un+mod%C3%A8le+de+scoring/Projet+Mise+en+prod+-+home-credit-default-risk.zip"
url = urllib.request.urlopen(path)
with zipfile.ZipFile(BytesIO(url.read())) as zfile:
    dfs = {name[:-4]: pd.read_csv(zfile.open(name), encoding='cp1252')
           for name in zfile.namelist()
           }
    zfile.close()




categorical_columns = []
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


def merging_data():
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
    return data,y

def prepare_data(random_state):
    data,y=merging_data()
    data.info(memory_usage='deep')
    columns = list(data.columns)
    x_tr, x_val, y_tr, y_val = train_test_split(data, y, test_size=0.2, random_state=random_state)
    y_val.sample(frac=0.1,random_state=42).to_csv(r'C:/Users/Utilisateur/OneDrive/Bureau/PROJET7/y_valid.csv', index=False)
    x_val.sample(frac=0.1,random_state=42).to_csv(path_or_buf=r'C:/Users/Utilisateur/OneDrive/Bureau/PROJET7/x_valid.csv')
    scaler = StandardScaler()
    scaler.fit(x_tr)
    x_tr = scaler.transform(x_tr)
    x_vali = scaler.transform(x_val)
    x_tr =x_tr.astype(np.float32)
    y_val =y_val.astype(np.float32)
    y_tr =y_tr.astype(np.float32)
    x_vali = x_vali.astype(np.float32)
    x_val_dash=scaler.transform(x_val.sample(frac=0.1, random_state=42))
    np.save(r'C:/Users/Utilisateur/OneDrive/Bureau/PROJET7/val', x_val_dash)
    return x_tr, x_vali, y_tr, y_val, columns, scaler, x_val, data, y


def mem_usage(pandas_obj):
    if isinstance(pandas_obj, pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else:
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2  # convertir les bytes en megabytes
    return "{:03.2f} MB".format(usage_mb)  # afficher sous format nombre (min 3 chiffres) et une précisionµ

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
