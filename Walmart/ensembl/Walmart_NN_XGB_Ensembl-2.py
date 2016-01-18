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

import theano
from lasagne import layers, nonlinearities
from nolearn.lasagne import NeuralNet, BatchIterator

from sklearn.linear_model import LogisticRegression


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
full_df.DepartmentDescription.fillna('UNKNOWN', inplace=True)
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

chi_sq_best = SelectKBest(score_func=chi2, k = 10000)
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

#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 5000, random_state = 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

y_train = y_train.astype('int32')

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)
        
class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()
            
def float32(k):
    return np.cast['float32'](k)

nn = NeuralNet(layers = [
     ('input', layers.InputLayer),
     ('dropout', layers.DropoutLayer),
     ('hidden1', layers.DenseLayer),
     ('dropout1', layers.DropoutLayer),   
     ('hidden2', layers.DenseLayer),
     ('dropout2', layers.DropoutLayer),   
     ('output', layers.DenseLayer),],
               
     input_shape = (None, X_train.shape[1]),
     dropout_p =.20,
               
     hidden1_num_units = 200,
     dropout1_p = .20,
     hidden2_num_units = 100,
     dropout2_p = .20,
               
     output_num_units = np.unique(y_train).shape[0],
     output_nonlinearity = nonlinearities.softmax,
               
     update_learning_rate=theano.shared(float32(0.01)),
     update_momentum=theano.shared(float32(0.9)),
    
     batch_iterator_train=BatchIterator(batch_size=1024),
               
     on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.01, stop=0.0001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        EarlyStopping(patience=25)
        ],

     regression = False,
     max_epochs = 200,
     verbose = True
      )

nn.fit(X_train,y_train)

X_, X_val, y_, y_val = train_test_split(X_train, y_train, test_size = 25000, random_state = 13)

del X_
del y_

xgb = xgboost.XGBClassifier(max_depth = 15, n_estimators = 200,
                        objective='multi:softprob', subsample = .80, colsample_bytree=.5)

xgb.fit(X_train, y_train, eval_set = [(X_val, y_val)], eval_metric = 'mlogloss', early_stopping_rounds=25)

y_xgb_train_predictions = xgb.predict_proba(X_train)
y_nn_train_predictions = nn.predict_proba(X_train)

X_ensembl_train = np.concatenate((y_xgb_train_predictions, y_nn_train_predictions), axis = 1)

y_nn_test_predictions = nn.predict_proba(X_test)
y_xgb_test_predictions = xgb.predict_proba(X_test)


col_names = ['TripType_' + str(c) for c in enc.classes_.astype('int')]

submission = pd.DataFrame(np.round(y_nn_test_predictions, 4), index=y_df[pd.isnull(y_df.TripType)].index, columns = col_names)
submission.reset_index(inplace = True)
submission.to_csv('Walmart_ensembl_NN_10000Features-Notebook.csv', index=False)

submission = pd.DataFrame(np.round(y_xgb_test_predictions, 4), index=y_df[pd.isnull(y_df.TripType)].index, columns = col_names)
submission.reset_index(inplace = True)
submission.to_csv('Walmart_ensembl_XGB_10000Features-Notebook.csv', index=False)

log_ensembl = LogisticRegression()
log_ensembl.fit(X_ensembl_train, y_train)
X_ensembl_test = np.concatenate((y_xgb_test_predictions, y_nn_test_predictions), axis = 1)
y_probas = log_ensembl.predict_proba(X_ensembl_test)

submission = pd.DataFrame(np.round(y_probas, 4), index=y_df[pd.isnull(y_df.TripType)].index, columns = col_names)
submission.reset_index(inplace = True)
submission.to_csv('Walmart_log_ensembl_10000Features-Notebook.csv', index=False)

y_probas_avg = (y_xgb_test_predictions + y_nn_test_predictions)/2

submission = pd.DataFrame(np.round(y_probas_avg,4), index=y_df[pd.isnull(y_df.TripType)].index, columns = col_names)
submission.reset_index(inplace = True)
submission.to_csv('Walmart_avg_ensembl_10000Features-Notebook.csv', index=False)
