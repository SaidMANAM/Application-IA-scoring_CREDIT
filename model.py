import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
import gc
import os
from sklearn import pipeline
from collections import Counter
import pickle
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_predict, \
    KFold, RandomizedSearchCV, cross_validate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

from sklearn.ensemble import RandomForestClassifier

from imblearn.combine import SMOTETomek

from Preprocessing import merging_data

from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import make_scorer


def cost_function(y_true, y_pred, **kwargs):
    global x
    cost = (((y_pred == 0) & (y_true == 0)) * x['AMT_CREDIT'] * (0.03)
            - ((y_pred == 1) & (y_true == 0)) * x['AMT_CREDIT'] * (0.03)
            - ((y_pred == 0) & (y_true == 1)) * x['AMT_CREDIT'] * (1 + 0.03))
    return np.sum(cost)


x_train, x_valid, y_train, y_valid, list_colonnes, std_scaler, x_exp, x, Y = merging_data(42)


# def data_balance(data, y):
#     count = Counter(y)
#     percent = {key: 100 * value / len(y) for key, value in count.items()}
#     print("avant de balancer le dataset, la répartiti   ondes   classes en  %   était:", percent)  # pourcentage
#     # data = reduce_mem_usage(data)
#     balancer = SMOTETomek(random_state=42)
#     data, y = balancer.fit_resample(data, y)
#     count = Counter(y)
#     percent = {key: 100 * value / len(y) for key, value in count.items()}
#     print("Les classes après  balance dataset:", percent)
#     return data, y


def performance(y, prediction):
    print("accuracy", accuracy_score(y, prediction))
    print("f1 score macro", f1_score(y, prediction, average='micro'))
    print("precision score", precision_score(y, prediction, average='micro'))
    print("recall score", recall_score(y, prediction, average='micro'))
    print("classification_report    \n", classification_report(y, prediction))


def random_classifier(random_state, x_tr, x_val, y_tr, y_val):
    dummy_clf = DummyClassifier(strategy="stratified", random_state=random_state).fit(x_tr, y_tr)
    predicted = cross_val_predict(dummy_clf, x_tr, y_tr)
    print("Performances en phase d'entrainement")
    performance(y_tr, predicted)
    predicted_valid = cross_val_predict(dummy_clf, x_val, y_val)
    print("Performances en phase de test")
    performance(y_val, predicted_valid)
    print("Training AUC&ROC", roc_auc_score(y_tr, dummy_clf.predict_proba(x_tr)[:, 1]))
    print("Testing AUC&ROC", roc_auc_score(y_val, dummy_clf.predict_proba(x_val)[:, 1]))
    del predicted
    gc.collect()
    return dummy_clf


def random_forest_classifier(random_state, x_tr, x_val, y_tr, y_val):
    over = SMOTE(random_state=42, sampling_strategy=0.1)
    under = RandomUnderSampler(sampling_strategy=0.5)
    steps = [('over', over), ('under', under), ('model', RandomForestClassifier(random_state=random_state))]
    pipe = Pipeline(steps=steps)
    cv = KFold(n_splits=3)

    # Number of trees in random forest
    n_estimators = np.linspace(start=10, stop=80, num=10, dtype=int)
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']

    # Maximum number of levels in tree
    max_depth = [2, 4]
    #     # Minimum number of samples required to split a node
    min_samples_split = [2, 5]
    #     # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2]
    #     # Method of selecting samples for training each tree
    bootstrap = [True, False]
    #
    #     # Create the param grid
    param_grid = {'model__n_estimators': n_estimators,
                  'model__max_features': max_features,
                  'model__max_depth': max_depth,
                  'model__min_samples_split': min_samples_split,
                  'model__min_samples_leaf': min_samples_leaf,
                  'model__bootstrap': bootstrap
                  }
    score = make_scorer(cost_function, greater_is_better=True)
    grid_cv = RandomizedSearchCV(estimator=pipe, param_distributions=param_grid, n_iter=25, cv=cv, scoring='roc_auc',
                                 random_state=random_state, n_jobs=-1, verbose=True, refit=True, error_score='raise')
    print('cross validation')
    grid_cv.fit(x_tr, y_tr)
    best_params = grid_cv.best_params_
    best_model = grid_cv.best_estimator_
    scores = cross_validate(best_model, x_tr, y=y_tr, scoring='roc_auc', cv=5, verbose=True, return_train_score=True,error_score='raise',
                            return_estimator=True)
    print('Train Area Under the Receiver Operating Characteristic Curve - : {:.3f} +/- {:.3f}'.format(
        scores['train_score'].mean(), scores['train_score'].std()))
    print('Validation Area Under the Receiver Operating Characteristic Curve - : {:.3f} +/- {:.3f}'.format(
        scores['test_score'].mean(), scores['test_score'].std()))
    print('Test Area Under the Receiver Operating Characteristic Curve - : {:.3f}'.format(
        roc_auc_score(y_val, best_model['model'].predict_proba(x_val)[:, 1])))
    print(best_model)
    del grid_cv,scores
    gc.collect()
    return best_params, best_model


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
    del gs, scores
    gc.collect()
    return best_params, best_model


def cost_threshold(y_true, y_pred, X):
    cost = (((y_pred == 0) & (y_true == 0)) * X['AMT_CREDIT'] * (0.03)
            - ((y_pred == 1) & (y_true == 0)) * X['AMT_CREDIT'] * (0.03)
            - ((y_pred == 0) & (y_true == 1)) * X['AMT_CREDIT'] * (1 + 0.03))
    return np.sum(cost)


def treshold(model, x_valid, exp, y_valid):
    seuils = []
    for value in np.arange(0, 1, 0.02):
        seuils.append(value)
    cout = []
    y_pred_proba = model.predict_proba(x_valid)
    y_ = y_pred_proba[:, 1]
    for elt in seuils:
        y_predected = np.array([1 if e > elt else 0 for e in y_])
        cout.append(cost_threshold(y_valid, y_predected, exp))
    res = seuils[np.argmax(cout)]
    return res


def pipeline_trained(model, scaler, X, y):
    pipeline_pred = pipeline.Pipeline([('scaler', scaler),
                                       ('model', model)])
    if X.shape[-1] == 770:
        X.set_index('SK_ID_CURR', inplace=True)
    pipeline_pred.fit(X, y)
    cwd = os.getcwd() + '\classifier.pkl'
    # pickle_out = open(r'C:/Users/Utilisateur/OneDrive/Bureau/PROJET7/classifier.pkl', "wb")
    pickle_out = open(cwd, "wb")
    pickle.dump(pipeline_pred, pickle_out)
    pickle_out.close()
    return "model entrainé et serialisé"


best_params, best_model = lightgbm_classifier(42, x_train, x_valid, y_train, y_valid)
pipeline_trained(best_model['model'], std_scaler, x, Y)

del x_train, x_valid, y_train, y_valid, list_colonnes, std_scaler, x_exp, x, Y, best_params, best_model
gc.collect()
