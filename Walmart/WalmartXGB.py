import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.externals import joblib
from sklearn.decomposition import RandomizedPCA

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder

import xgboost

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

train_y = train.loc[:, ['VisitNumber', 'TripType']]
train_y.drop_duplicates('VisitNumber', inplace=True)

train.drop(['TripType'], axis = 1, inplace = True)

train.Upc.fillna(-100, inplace=True)
train.DepartmentDescription.fillna('Unknown', inplace=True)
train.FinelineNumber.fillna(-100, inplace=True)

test.Upc.fillna(-100, inplace=True)
test.DepartmentDescription.fillna('Unknown', inplace=True)
test.FinelineNumber.fillna(-100, inplace=True)

train['FinelineNumber'] = train['FinelineNumber'].astype('int')
test['FinelineNumber'] = test['FinelineNumber'].astype('int')

train['DeptItems'] = train.DepartmentDescription +' ' + train.FinelineNumber.astype('str')
test['DeptItems'] = test.DepartmentDescription +' ' + test.FinelineNumber.astype('str')

full_df = pd.concat((train, test))
full_df_pivot = pd.pivot_table(full_df, values='ScanCount', index='VisitNumber',columns='DeptItems')
full_df_pivot.fillna(0, inplace=True)

X_full = full_df_pivot.values

pca = joblib.load('pca.pkl')

X = full_df_pivot.loc[train_y.VisitNumber].values
y = train_y['TripType'].values

X_transformed = pca.transform(X)

enc = LabelEncoder()
y = enc.fit_transform(y)

X_train, X_val, y_train, y_val = train_test_split(X_transformed, y, test_size = 5000, random_state = 1)

test_visits = test.VisitNumber
test_visits.drop_duplicates(inplace = True)
X_test = full_df_pivot.loc[test_visits,:].values
X_test_transformed = pca.transform(X_test)

del X_full
del full_df_pivot

xgb = xgboost.XGBClassifier(max_depth = 10, n_estimators = 500, silent=False,
                        objective='multi:softprob', subsample = .9, colsample_bytree=.8)

xgb.fit(X_train, y_train, eval_set = [(X_val, y_val)], eval_metric = 'mlogloss', early_stopping_rounds=25)

y_probas = xgb.predict_proba(X_test_transformed)
col_names = ['TripType_' + str(c) for c in enc.classes_]

submission = pd.DataFrame(y_probas, index=test_visits, columns = col_names)
submission.reset_index(inplace = True)

submission.to_csv('Walmart_submission_XGB_500Iterations.csv', index=False)




