import sys
sys.path.append('../core')

import numpy as np
import pandas as pd

#data prep
from load_data import prep_data

#modeling utilities
from utilities import modeModel
from utilities import undersample
from utilities import oversample
from utilities import smote

#visualizations
from visualization import roc_plot
import matplotlib.pyplot as plt

#import modeling packages
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools import add_constant
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

def mode_cross_val(X, y):
    #function takes in endog and exog variables, does a logistic regression
    #returns mean accuracy, precision and recall

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


def logit_cross_val(X, y):
    #function takes in endog and exog variables, does a logistic regression
    #returns mean accuracy, precision and recall

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


def logit_pred(X,y, prob=False):
    #function takes in endog and exog variables, does a logistic regression
    #returns predictions from the model
    model = LogisticRegression()
    model.fit(X, y)
    if prob==True:
        return model.predict_proba(X)[:, 1]
    else:
        return model.predict(X)


if __name__ == '__main__':

    df = pd.read_csv('/Users/etownbetty/Documents/Galvanize/Project/data/purchaseCustomerCut7DataTrainingSet.csv')

    y = df['repeat'].values
    X = df.drop(['repeat', 'CustomerNo'], axis=1).values
    #smote the data
    X_smote, y_smote = smote(X, y, 0.5)
    X_const = add_constant(X_smote, prepend=True)

    logit_model = Logit(y_smote, X_const).fit()
    logit_model.summary()
    y_predict = logit_model.predict(X_const)

    #check a baseline model that is just the mode assigned to each indivs
    mode_model_acc, mode_model_precision, mode_model_recall = mode_cross_val(X_smote, y_smote)
    print("ModelAccuracy: {}, ModelPrecision: {}, ModelRecall: {}".format(mode_model_acc, mode_model_precision, mode_model_recall))

    model_acc, model_precision, model_recall = logit_cross_val(X_smote, y_smote)
    print("ModelAccuracy: {}, ModelPrecision: {}, ModelRecall: {}".format(model_acc, model_precision, model_recall))

    y_smote_predict = logit_pred(X_smote, y_smote, prob=True)
    logit_fpr, logit_tpr, logit_thresholds = roc_curve(y_smote, y_smote_predict)
    logit_auc = roc_auc_score(y_smote, y_smote_predict)

    #fit ROC curve
    roc_plot(logit_fpr, logit_tpr, "False Positive Rate (%)", "True Positive Rate (%)", "ROC Curve for Logit Model, AUC: {}".format(logit_auc), "LOGIT model", 'LogisticRegressionROCplot.png')
