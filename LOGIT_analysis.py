import sys
sys.path.append('src')

import numpy as np
import pandas as pd

#modeling utilities
from utilities import modeModel
from utilities import undersample
from utilities import oversample
from utilities import smote

#visualizations
from visualization import roc_plot
import matplotlib.pyplot as plt

#import modeling packages
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools import add_constant
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, roc_auc_score

class logitReg(object):

    def __init__(self, df):
        self.df = df

    def mode_cross_val(self, X, y):
        '''
        Takes in X and y, does KFold cross-Val for a mode model of logistic regression
        Returns accuracy, precision and recall
        '''
        kfold = StratifiedKFold(y, 3)

        accuracies = []
        precisions = []
        recalls = []

        for train_index, test_index in kfold:
            model = modeModel()
            model.fit(y[train_index])
            y_predict = model.predict(X[test_index])
            y_true = y[test_index]
            accuracies.append(accuracy_score(y_true, y_predict))
            precisions.append(precision_score(y_true, y_predict))
            recalls.append(recall_score(y_true, y_predict))

        return np.average(accuracies), np.average(precisions), np.average(recalls)


    def logit_cross_val(self, X, y):
        '''
        Takes in X and y, does KFold cross-Val for logistic regression
        Returns accuracy, precision and recall
        '''

        kfold = StratifiedKFold(y, 3)

        accuracies = []
        precisions = []
        recalls = []

        for train_index, test_index in kfold:
            model = LogisticRegression()
            model.fit(X[train_index], y[train_index])
            y_predict = model.predict(X[test_index])
            y_true = y[test_index]
            accuracies.append(accuracy_score(y_true, y_predict))
            precisions.append(precision_score(y_true, y_predict))
            recalls.append(recall_score(y_true, y_predict))

        return np.average(accuracies), np.average(precisions), np.average(recalls)

    def pred_logit(self, X,y, prob=False):
        '''
        Takes in X and y, does logstic regression and outputs predictions or predicted probabilities
        '''
        LR = LogisticRegression()
        LR.fit(X, y)
        if prob==True:
            return LR.predict_proba(X)[:, 1]
        else:
            return LR.predict(X)

    def fit_logit(self):
        '''
        Takes in DF and does logistic regression for X vs Y
        Prints out baseline mode model diagnostics and predicted model diagnostics and ROC curve
        Returns SMOTE X and y values
        '''
        self.y = self.df['repeat'].values
        self.X = self.df.drop(['repeat', 'CustomerNo'], axis=1).values
        #smote the data
        self.X_smote, self.y_smote = smote(self.X, self.y, 0.5)
        self.X_const = add_constant(self.X_smote, prepend=True)
        logit_model = Logit(self.y_smote, self.X_const).fit()
        print(logit_model.summary())
        y_predict = logit_model.predict(self.X_const)

        #check a baseline model that is just the mode assigned to each indivs
        mode_model_acc, mode_model_precision, mode_model_recall = self.mode_cross_val(self.X_smote, self.y_smote)
        print("ModelAccuracy: {}, ModelPrecision: {}, ModelRecall: {}".format(mode_model_acc, mode_model_precision, mode_model_recall))

        model_acc, model_precision, model_recall = self.logit_cross_val(self.X_smote, self.y_smote)
        print("ModelAccuracy: {}, ModelPrecision: {}, ModelRecall: {}".format(model_acc, model_precision, model_recall))

        return self.X_smote, self.y_smote

    def diag_logit(self, y_smote, y_pred, roc_file):
        '''
        Plots an ROC curve, and does auc/roc diagnostics
        '''
        logit_fpr, logit_tpr, logit_thresholds = roc_curve(y_smote, y_pred)
        logit_auc = roc_auc_score(y_smote, y_pred)
        #fit ROC curve
        roc_plot(logit_fpr, logit_tpr, "False Positive Rate (%)", "True Positive Rate (%)", "ROC Curve for Logit Model, AUC: {}".format(logit_auc), "LOGIT model", roc_file)

if __name__ == '__main__':

    df = pd.read_csv('/Users/etownbetty/Documents/Galvanize/Project/data/purchaseCustomerCut7DataTrainingSet.csv')

    LR = logitReg(df)

    X_smote, y_smote = LR.fit_logit()

    y_pred = LR.pred_logit(X_smote, y_smote, prob=True)

    LR.diag_logit(y_smote, y_pred, 'LogisticRegressionROCplot.pdf')
