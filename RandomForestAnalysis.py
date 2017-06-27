import sys

import pandas as pd
import numpy as np

#import utilities
from src.utilities import smote
from src.utilities import get_scores

#import model statements
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.cross_validation import train_test_split
from visualization import roc_plot
from visualization import plot_roc

#import plotting
import matplotlib.pyplot as plt

class RFmodel(object):

    def load_data(self, filepath, dates=None, yr_first=False):
        '''
        Load data from csv file located in the data folder
        '''
        if dates:
            df = pd.read_csv(filepath, parse_dates=dates, date_parser = pd.tseries.tools.to_datetime)
        else:
            df = pd.read_csv(filepath)
        self.df = df
        return df

    def prepare_data(self, filepath, dates=None, yr_first=False):
        '''
        Make X and y, smote values to 0.5
        '''
        self.df = self.load_data(filepath)
        #make new variables
        y = self.df['repeat'].values
        X = self.df.drop(['repeat', 'CustomerNo'],axis=1).values
        #smote values to 0.5
        self.X_smote, self.y_smote = smote(self.X, self.y, 0.5)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_smote, self.y_smote)
        self.cols = self.df.drop('price', axis=1).columns

    def rf_cross_val(self, X, y):
        '''
        Takes in X and y, does 3 fold stratified Kfold cross validation
        Returns average accuracy, precision and recall
        '''
        kfold = StratifiedKFold(y, 3)

        accuracies = []
        precisions = []
        recalls = []

        for train_index, test_index in kfold:
            model = RandomForestClassifier()
            model.fit(X[train_index], y[train_index])
            y_predict = model.predict(X[test_index])
            y_true = y[test_index]
            accuracies.append(accuracy_score(y_true, y_predict))
            precisions.append(precision_score(y_true, y_predict))
            recalls.append(recall_score(y_true, y_predict))

        return np.average(accuracies), np.average(precisions), np.average(recalls)

    def rf_fit(self):
        '''
        Does a preliminary rf cross validation, fits model with the X, y training and returns feature importances
        '''
        # smote values and print out preliminary model accuracy, precision and recall
        model_acc, model_precision, model_recall = self.rf_cross_val(self.X_smote, self.y_smote)
        print("ModelAccuracy: {}, ModelPrecision: {}, ModelRecall: {}".format(model_acc, model_precision, model_recall))

        #fit model to RandomForestClassifier, find most important features, get some diagnostic plots
        #make a test-train split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_smote, self.y_smote)

        RF = RandomForestClassifier()
        forest_fit = RF.fit(self.X_train, self.y_train)
        y_predict = RF.predict(self.X_test)

        print("Random Forest Confusion Matrix")
        print(confusion_matrix(self.y_test, y_predict))
        self.RF.feature_importances_
        self.RF.estimators_

    def rf_feature_diagnostics(self):
        #print out feature importances
        sorted_importances = np.argsort(self.RF.feature_importances_)
        print("top five feature importances:", list(df.drop(['repeat', 'CustomerNo'],axis=1).columns[sorted_importances[-1:-6:-1]]))

        n = 10 # top 10 features
        importances = self.RF.feature_importances_[:n]
        std = np.std([tree.feature_importances_ for tree in self.RF.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]

        # Print the feature ranking
        print("Feature ranking:")
        features = list(self.df.drop(['repeat', 'CustomerNo'],axis=1).columns[self.RF.feature_importances_[-1:-11:-1]])
        for f in range(n):
            print("%d. %s (%f)" % (f + 1, features[f], importances[indices[f]]))

        # Plot the feature importances of the forest
        fig, ax = plt.subplots(1)
        plt.title("Feature importances")
        plt.bar(range(10), importances[indices], yerr=std[indices], color="r", align="center")
        plt.xticks(range(10))
        plt.xlim([-1, 10])
        textstr = ""
        for i in range(10):
            n = i+1
            textstr = "".join([textstr, "{} : {}\n".format(n, features[i])])
        ax.text(0.6, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
        ax.set_xticklabels(range(1,11))
        plt.savefig("RandomForest_feature_importances_barplot.png")
        plt.close()

    def rf_vary_trees(self):
        '''
        Investigates optimal number of trees in the forest, plots accuracy
        '''
        num_trees = range(50, 500, 50)
        accuracies = []
        for n in num_trees:
            tot = 0
            for i in range(5):
                rf = RandomForestClassifier(n_estimators=n)
                rf.fit(self.X_train, self.y_train)
                tot += rf.score(self.X_test, self.y_test)
            accuracies.append(tot / 5)
        plt.plot(num_trees, accuracies)
        plt.xlabel("number of trees")
        plt.ylabel("accuracy_score")
        plt.title("Number of RF trees Vs Accuracy Score")
        plt.savefig("NumberRFtreesVsAccuracyScore.png")
        plt.close()

    def rf_vary_params(self):
        '''
        Investigates optimal number of parameters in the model, plots accuracy
        '''
        num_features = range(1, len(self.df.drop(['repeat', 'CustomerNo'],axis=1).columns) + 1)
        max_feature_accuracies = []
        for n in num_features:
            tot = 0
            for i in range(5):
                rf = RandomForestClassifier(max_features=n)
                rf.fit(self.X_train, self.y_train)
                tot += rf.score(X_test, y_test)
            max_feature_accuracies.append(tot / 5)
        plt.plot(num_features, max_feature_accuracies)
        plt.xlabel("number of features")
        plt.ylabel("accuracy_score")
        plt.title("Number of RF features Vs Accuracy Score")
        plt.savefig("NumberRFfeaturesVsAccuracyScore.png")
        plt.close()

    def roc_plots(self):
        print("    Random Forest:", get_scores(RandomForestClassifier, self.X_train, self.X_test, self.y_train, self.y_test, n_estimators=25, max_features=5))
        print("    Logistic Regression:", get_scores(LogisticRegression, self.X_train, self.X_test, self.y_train, self.y_test))
        print("    Decision Tree:", get_scores(DecisionTreeClassifier, self.X_train, self.X_test, self.y_train, self.y_test))

        plot_roc(self.X_train, self.y_train, RandomForestClassifier, "RF_ROC_plot.png", n_estimators=25, max_features=5)
        plt.close()
        plot_roc(self.X_train, self.y_train, LogisticRegression, "LogisticRegression_ROC_plot.png")
        plt.close()
        plot_roc(self.X_train, self.y_train, DecisionTreeClassifier, "DecisionTreeClassifier_ROC_plot.png")
        plt.close()

if __name__=="__main__":
    '''
    read in 1, 2 purchase customer data with less than 7 purchases
    '''
    df_file = sys.argv[1]
    rf = RFmodel()
    df = rf.prepare_data(df_file)
    #fit model and print out
    rf.rf_fit()
    #prints out feature importances and plots top 10
    rf.rf_feature_diagnostics()
    #investigates optimal number of trees
    rf.rf_vary_trees()
    ## Levels off around 20-25
    #investigate the max features parameter
    rf.rf_vary_params()
    # Run all the other relevant classifiers
    rf.roc_plots()
