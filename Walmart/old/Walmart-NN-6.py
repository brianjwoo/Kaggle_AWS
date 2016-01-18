import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.externals import joblib

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.metrics import confusion_matrix

import xgboost

train = pd.read_csv('./train.csv') #Last visit number is 191347
test = pd.read_csv('./test.csv') #Last visit number is 191348

full_df = pd.concat((train, test))

full_df_negatives = full_df[full_df.ScanCount < 0]
full_df_negatives_agg = full_df_negatives.groupby(['VisitNumber']).agg({'ScanCount':np.sum}) #Negative Feature Count

full_df_uncategorized = full_df[pd.isnull(full_df.Upc)]
full_df_uncategorized_agg = full_df_uncategorized.groupby(['VisitNumber']).agg({'ScanCount':np.sum}) #Unknown Feature Count

full_df_totals = full_df[full_df.ScanCount > 0]
full_df_totals_agg = full_df_totals.groupby(['VisitNumber']).agg({'ScanCount':np.sum}) #Total purchases Feature Count


full_df.Upc.fillna(-100, inplace=True)
full_df.DepartmentDescription.fillna('Unknown', inplace=True)
full_df.FinelineNumber.fillna(-100, inplace=True)


visit_days = full_df.loc[:,['VisitNumber','Weekday']]
visit_days.drop_duplicates('VisitNumber', inplace = True)
visit_days.set_index('VisitNumber', inplace = True)
visit_days = pd.get_dummies(visit_days)

full_df['FinelineNumber'] = full_df['FinelineNumber'].astype('int')
full_df['DeptItems'] = full_df.DepartmentDescription +' ' + full_df.FinelineNumber.astype('str')

full_deptitems_df = pd.pivot_table(full_df[full_df.ScanCount>0], values='ScanCount', index='VisitNumber',columns='DeptItems', aggfunc=np.sum)
full_deptitems_df.fillna(0, inplace=True)


y_df = full_df.loc[:, ['VisitNumber', 'TripType']]
y_df.drop_duplicates('VisitNumber', inplace=True)
y_df.set_index('VisitNumber', inplace=True)

y_df = y_df.join(full_deptitems_df) #This requires an insane amount of memory **Cannot fill 0s due to memory error

del full_deptitems_df

X_train = y_df[pd.notnull(y_df.TripType)].drop('TripType', axis = 1).values
X_test = y_df[pd.isnull(y_df.TripType)].drop('TripType', axis = 1).values
y_train = y_df[pd.notnull(y_df.TripType)]['TripType'].values


y_df = y_df[['TripType']] #Removing Unneccessary Columns


X_train = np.nan_to_num(X_train) #Splitting this into 2 cells works

chi_sq_best = SelectKBest(score_func=chi2, k = 7000)
chi_sq_best.fit(X_train,y_train)

X_train = chi_sq_best.transform(X_train)

X_test = np.nan_to_num(X_test)
X_test = chi_sq_best.transform(X_test)

X_df = pd.pivot_table(full_df, values='ScanCount', index='VisitNumber',columns='DepartmentDescription', aggfunc=np.sum)
X_df.fillna(0, inplace=True)


X_df = X_df.join(full_df_totals_agg, rsuffix='Totals')
X_df = X_df.join(full_df_uncategorized_agg, rsuffix='Uncategorized')
X_df = X_df.join(full_df_negatives_agg, rsuffix='Negatives')
X_df = X_df.join(visit_days)
X_df.fillna(0, inplace = True)

y_df = y_df.join(X_df)

X_train2 = y_df[pd.notnull(y_df.TripType)].drop('TripType', axis = 1).values
X_test2 = y_df[pd.isnull(y_df.TripType)].drop('TripType', axis = 1).values
y_train2 = y_df[pd.notnull(y_df.TripType)]['TripType'].values

X_train = np.concatenate((X_train, X_train2), axis = 1)
X_test = np.concatenate((X_test, X_test2), axis = 1)

enc = LabelEncoder()
y_train = enc.fit_transform(y_train)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 4000, random_state = 1)

xgb = xgboost.XGBClassifier(max_depth = 13, n_estimators = 200,
                        objective='multi:softprob', subsample = .80, colsample_bytree=.5)

xgb.fit(X_train, y_train, eval_set = [(X_val, y_val)], eval_metric = 'mlogloss', early_stopping_rounds=25)

y_probas = xgb.predict_proba(X_test)


col_names = ['TripType_' + str(c) for c in enc.classes_.astype('int')]
submission = pd.DataFrame(np.round(y_probas, 4), index=y_df[pd.isnull(y_df.TripType)].index, columns = col_names)

submission.reset_index(inplace = True)
submission.to_csv('Walmart_submission_XGB_7000Features-6.csv', index=False)


#cm = confusion_matrix(y_train,y_pred)
#cm_df = pd.DataFrame(cm, index = enc.classes_, columns=enc.classes_)
#cm_df.to_csv('Walmart_Confusion_Matrix-6.csv')

