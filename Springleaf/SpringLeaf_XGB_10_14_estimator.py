import pandas as pd
import numpy as np
import random
import time
import datetime

from sklearn.preprocessing import LabelEncoder
from sklearn.grid_search import RandomizedSearchCV
import sklearn.preprocessing
import sklearn.cross_validation
import sklearn.externals

import xgboost as xgb

train = pd.read_csv('./train.csv').set_index("ID")
test = pd.read_csv('./test.csv').set_index("ID")

drop_cols = [c for c in train.columns if len(np.unique(train[c])) == 1]#Should later drop columns from train and test

train.drop(drop_cols, axis = 1, inplace = True)
test.drop(drop_cols, axis = 1, inplace = True)

object_cols = train.columns[train.dtypes == 'object']
dates_cols =['VAR_0073', 'VAR_0075', 'VAR_0156', 'VAR_0157', 'VAR_0158', 'VAR_0159', 'VAR_0166', 'VAR_0167', 'VAR_0168', 'VAR_0169', 'VAR_0176', 'VAR_0177', 'VAR_0178','VAR_0179', 'VAR_0204', 'VAR_0217']

redundant_0008 = ['VAR_0008', 'VAR_0009', 'VAR_0010','VAR_0011','VAR_0012', 'VAR_0043','VAR_0044','VAR_0196', 'VAR_0229', ] #Contains redundant information
train.drop(redundant_0008, axis = 1, inplace = True)
test.drop(redundant_0008, axis = 1, inplace = True)

train['NaN'] = 0
test['NaN'] = 0

for c in test.columns: #Engineered Feature #1
    train['NaN'] += pd.isnull(train[c])
    test['NaN'] += pd.isnull(test[c])

def to_date(date_str):
    if pd.notnull(date_str):
        return datetime.datetime.strptime(date_str, '%d%b%y:%H:%M:%S')
    else:
        return date_str

for c in dates_cols:
    train[c] = train[c].apply(to_date)
    test[c] = test[c].apply(to_date)


diff = list(set(test.columns[test.dtypes == 'datetime64[ns]']) - set(train.columns[train.dtypes == 'datetime64[ns]'])) ##test got converted into date time format while train is still object format

train.drop(diff, axis = 1, inplace = True)
test.drop(diff, axis = 1, inplace = True)

dates_cols_fixed = train.columns[train.dtypes == 'datetime64[ns]']

train['VAR_0204-VAR_0217'] = (train['VAR_0204'] - train['VAR_0217']).apply(lambda x: x.astype('timedelta64[D]')/np.timedelta64(1, 'D'))
test['VAR_0204-VAR_0217'] = (test['VAR_0204'] - test['VAR_0217']).apply(lambda x: x.astype('timedelta64[D]')/np.timedelta64(1, 'D'))

train['VAR_0204-VAR_0075'] = (train['VAR_0204'] - train['VAR_0075']).apply(lambda x: x.astype('timedelta64[D]')/np.timedelta64(1, 'D'))
test['VAR_0204-VAR_0075'] = (test['VAR_0204'] - test['VAR_0075']).apply(lambda x: x.astype('timedelta64[D]')/np.timedelta64(1, 'D'))


train['VAR_0073_ISNULL'] = pd.isnull(train['VAR_0073']).astype('int')
test['VAR_0073_ISNULL'] = pd.isnull(test['VAR_0073']).astype('int')

errors_dates = [] #Does not matter because will convert to float
for c in dates_cols_fixed: 
    try:
        train[c+'_YEAR'] = train[c].apply(lambda x: x.year)
        test[c+'_YEAR'] = test[c].apply(lambda x: x.year)
        train[c+'_MONTH'] = train[c].apply(lambda x: x.month)
        test[c+'_MONTH'] = test[c].apply(lambda x: x.month)
    except:
        errors_dates.append(c)

train.drop(dates_cols_fixed, axis = 1, inplace=True)
test.drop(dates_cols_fixed, axis = 1, inplace=True)

object_cols = train.columns[train.dtypes == 'object']

unfixed_dates = ['VAR_0157','VAR_0167','VAR_0177', 'VAR_0158']   

train.loc[:,unfixed_dates] = pd.isnull(train[unfixed_dates]).astype('int') #Added 10-14
test.loc[:,unfixed_dates] = pd.isnull(test[unfixed_dates]).astype('int')

#train.drop(unfixed_dates, axis=1, inplace = True)
#test.drop(unfixed_dates, axis=1, inplace = True)

object_cols = train.columns[train.dtypes == 'object']

train.loc[:,object_cols] = train[object_cols].fillna(-10)
test.loc[:,object_cols] = test[object_cols].fillna(-10)

object_cols = train.columns[train.dtypes == 'object']

encoders = {}
errors = []
for col in object_cols:
    if col not in dates_cols:
        encoders[col] = LabelEncoder()
        encoders[col].fit(pd.concat((train[col], test[col])))
        try:
            train[col] = encoders[col].transform(train[col])
            test[col] = encoders[col].transform(test[col])
        except:
            errors.append(col)

object_cols = train.columns[train.dtypes == 'object']

train.drop(object_cols, axis=1, inplace = True)
test.drop(object_cols, axis=1, inplace = True)

X = train.drop('target', axis = 1).values
y = train.target.values
X_test = test.values

imputer = sklearn.preprocessing.Imputer(strategy='median')
X = imputer.fit_transform(X)
X_test = imputer.transform(X_test)

X_train, X_val, y_train, y_val = sklearn.cross_validation.train_test_split(X, y, random_state = 42, test_size = .05)

xgb_clf = xgb.XGBClassifier(max_depth=20, learning_rate=.020, n_estimators=3000, subsample=.9, \
                             colsample_bytree=.6)

xgb_clf.fit(X_train, y_train, eval_set = [(X_val, y_val)], eval_metric='auc', early_stopping_rounds=20)
y_pred = xgb_clf.predict_proba(X_test)


##GRID SEARCH
#xgb_clf = xgb.XGBClassifier()
#grid_params = {'max_depth':[14, 20],'learning_rate':[.020], \
#               'n_estimators':[2500], 'subsample':[.9], 'colsample_bytree':[.6]}
#
#rand_gridsearch = RandomizedSearchCV(xgb_clf, param_distributions = grid_params, \
#                                     n_iter = 6, scoring = 'roc_auc', cv = 3, verbose = True, random_state = 42) #Changed to 3 Fold CV
#
#rand_gridsearch.fit(X,y)

#print '======================================================='
#print rand_gridsearch.best_params_
#print '======================================================='
#
#
#for s in rand_gridsearch.grid_scores_:
#    print s
#
#best_xgb = rand_gridsearch.best_estimator_ 
#
#y_pred = rand_gridsearch.predict_proba(X_test)
##

submission = pd.DataFrame(y_pred[:,1], index=test.index, columns=['target'])
submission.index.name = 'ID'
submission.to_csv('B_XGB_GridSearch_2015_10_14_estimator.csv')


#sklearn.externals.joblib.dump(best_xgb, './models/XGB_2015_10_14.pkl')
#sklearn.externals.joblib.dump(encoders, './models/XGB2_encoders_2015_10_14.pkl')





