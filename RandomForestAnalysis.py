import sys

import pandas as pd
import numpy as np

#import utilities
from src.utilities import smote
from src.utilities import get_scores

#import model statements
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.cross_validation import train_test_split
from visualization import roc_plot
from visualization import plot_roc

#import plotting
import matplotlib.pyplot as plt

def RF_cross_val(X, y):
    #function takes in endog and exog variables, does a logistic regression
    #returns mean accuracy, precision and recall

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

if __name__=="__main__":
    '''read in 1, 2 purchase customer data with less than 7 purchases

    '''
    df_file = sys.argv[1]
    df = pd.read_csv(df_file)

    #throw it into a random Forest
    y = df['repeat'].values
    X = df.drop(['repeat', 'CustomerNo'],axis=1).values

    #smote the values
    X_smote, y_smote = smote(X, y, 0.5)

    model_acc, model_precision, model_recall = RF_cross_val(X_smote, y_smote)
    print("ModelAccuracy: {}, ModelPrecision: {}, ModelRecall: {}".format(model_acc, model_precision, model_recall))

    #fit model to RandomForestClassifier, find most important features, get some diagnostic plots
    #make a test-train split
    X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote)

    RF = RandomForestClassifier()
    forest_fit = RF.fit(X_train, y_train)
    y_predict = RF.predict(X_test)

    print(confusion_matrix(y_test, y_predict))

    #print out feature importances
    feature_importances = np.argsort(RF.feature_importances_)
    print("top five feature importances:", list(df.drop(['repeat', 'CustomerNo'],axis=1).columns[feature_importances[-1:-6:-1]]))

    n = 10 # top 10 features
    importances = RF.feature_importances_[:n]
    std = np.std([tree.feature_importances_ for tree in RF.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")
    features = list(df_sideDataCut_7.drop(['repeat', 'CustomerNo'],axis=1).columns[feature_importances[-1:-11:-1]])
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

    # 14. Try modifying the number of trees
    num_trees = range(5, 50, 5)
    accuracies = []
    for n in num_trees:
        tot = 0
        for i in range(5):
            rf = RandomForestClassifier(n_estimators=n)
            rf.fit(X_train, y_train)
            tot += rf.score(X_test, y_test)
        accuracies.append(tot / 5)
    plt.plot(num_trees, accuracies)
    plt.xlabel("number of trees")
    plt.ylabel("accuracy_score")
    plt.title("Number of RF trees Vs Accuracy Score")
    plt.savefig("NumberRFtreesVsAccuracyScore.png")
    plt.close()
    ## Levels off around 20-25

    # 15. Try modifying the max features parameter
    num_features = range(1, len(df.drop(['repeat', 'CustomerNo'],axis=1).columns) + 1)
    max_feature_accuracies = []
    for n in num_features:
        tot = 0
        for i in range(5):
            rf = RandomForestClassifier(max_features=n)
            rf.fit(X_train, y_train)
            tot += rf.score(X_test, y_test)
        max_feature_accuracies.append(tot / 5)
    plt.plot(num_features, max_feature_accuracies)
    plt.xlabel("number of features")
    plt.ylabel("accuracy_score")
    plt.title("Number of RF features Vs Accuracy Score")
    plt.savefig("NumberRFfeaturesVsAccuracyScore.png")
    plt.close()
    ## Levels off around 5-6

    # Run all the other relevant classifiers that we have learned so far in class

    print("    Random Forest:", get_scores(RandomForestClassifier, X_train, X_test, y_train, y_test, n_estimators=25, max_features=5))
    print("    Logistic Regression:", get_scores(LogisticRegression, X_train, X_test, y_train, y_test))
    print("    Decision Tree:", get_scores(DecisionTreeClassifier, X_train, X_test, y_train, y_test))

    plot_roc(X, y, RandomForestClassifier, "RF_ROC_plot.png", n_estimators=25, max_features=5)
    plt.close()
    plot_roc(X, y, LogisticRegression, "LogisticRegression_ROC_plot.png")
    plt.close()
    plot_roc(X, y, DecisionTreeClassifier, "DecisionTreeClassifier_ROC_plot.png")
    plt.close()
