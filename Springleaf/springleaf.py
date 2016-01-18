import pandas as pd
import numpy as np
import random
from collections import Counter
import matplotlib.pyplot as plt
import time
import datetime

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.grid_search import GridSearchCV
import sklearn.preprocessing
import sklearn.cross_validation
import cPickle as pickle


train = pd.read_csv('./train.csv').set_index("ID")
test = pd.read_csv('./test.csv').set_index("ID")

drop_cols = [c for c in train.columns if len(np.unique(train[c])) == 1] #Dropping non-unique columns from dataset
train.drop(drop_cols, axis = 1, inplace = True)
test.drop(drop_cols, axis = 1, inplace = True)

object_cols = train.columns[train.dtypes == 'object']
dates_cols =['VAR_0073', 'VAR_0075', 'VAR_0156', 'VAR_0157', 'VAR_0158', 'VAR_0159', 'VAR_0166', 'VAR_0167', 'VAR_0168', 'VAR_0169', 'VAR_0176', 'VAR_0177', 'VAR_0178','VAR_0179', 'VAR_0204', 'VAR_0217']

redundant_0008 = ['VAR_0008', 'VAR_0009', 'VAR_0010','VAR_0011','VAR_0012', 'VAR_0043','VAR_0044','VAR_0196', 'VAR_0229', ] #Contains redundant information
train.drop(redundant_0008, axis = 1, inplace = True)
test.drop(redundant_0008, axis = 1, inplace = True)

train['NaNs'] = train.apply(lambda x: np.sum(pd.isnull(x)), axis = 1) #Count the number of NaNs in each row will allow us to separate the types of data
test['NaNs'] = test.apply(lambda x: np.sum(pd.isnull(x)), axis = 1)

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

errors_dates = [] #Does not matter because will convert to float
for c in dates_cols_fixed: 
    try:
        train[c] = train[c].apply(lambda x: x.year)
        test[c] = test[c].apply(lambda x: x.year)
    except:
        errors.append(c)

object_cols = train.columns[train.dtypes == 'object']

enc= {} #Generates encoders for remaining categorical variables
errors = []
for c in object_cols:
    try:
        enc[c] = LabelEncoder()
        train[c] = enc[c].fit_transform(train[c])
        test[c] = enc[c].transform(test[c])
    except:
        errors.append(c)

train.drop(errors, axis = 1, inplace = True) #Drops vars that were not correctly encoded probably due to not being present in training... can bind together and fix
test.drop(errors, axis = 1, inplace = True)

X = train.drop('target', axis = 1).values
y = train.target.values


imputer = sklearn.preprocessing.Imputer(strategy='median')
X = imputer.fit_transform(X)

rf = RandomForestClassifier(n_estimators = 45)
print sklearn.cross_validation.cross_val_score(rf, X, y, scoring = 'roc_auc')

ada = AdaBoostClassifier()
print sklearn.cross_validation.cross_val_score(ada, X, y, scoring = 'roc_auc')

rf.fit(X,y)

X_test = test.values
X_test = imputer.transform(X_test)

y_pred = rf.predict(X_test)


# MAKING SUBMISSION
submission = pd.DataFrame(y_pred, index=test.index, columns=['target'])
submission.index.name = 'ID'
submission.to_csv('beat_withrf_B.csv')


