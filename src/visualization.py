#this houses all visualization tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import NMF

############################################
#### EDA plots                          ####
####                                    ####
############################################

def hist(df, column, xlab, ylab, title, bins=20, savefig=None):
    #take in pandas dataframe
    #prints out or saves histogram of column, axes labeled
    if savefig:
        plt.hist(df[column], bins=bins)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.title(title)
        plt.savefig(savefig)
    else:
        plt.hist(df[column], bins=bins)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.title(title)
        plt.show()

def scatter(df, x_lab, y_lab, title, savefig=None):
    #takes in pandas dataframe, and colum
    #prints out or saves scatter plot of column
    if savefig:
        plt.scatter(df[x_lab], df[y_lab])
        plt.xlabel(x_lab)
        plt.ylabel(y_lab)
        plt.title(title)
        plt.savefig(savefig)
    else:
        plt.scatter(df[x_lab], df[y_lab])
        plt.xlabel(x_lab)
        plt.ylabel(y_lab)
        plt.title(title)
        plt.show()

############################################
####  RF plots                          ####
####                                    ####
############################################

# ranking customers for return business-
# ROC curve for purchase probability, total share of actual purchases in data set

def purchase_roc_curve(probabilities, purchases):
    '''
    INPUT: numpy array, numpy array
    OUTPUT: list, list

    Take a numpy array of the predicted probabilities and a numpy array of the
    true labels.
    Return the True Positive Rates, False Positive Rates and Thresholds for the
    ROC curve.
    '''

    thresholds = np.sort(probabilities)
    sorted_purchases = purchases[probabilities.argsort()]
    cum_purchases = np.cumsum(sorted_purchases)

    return thresholds, cum_purchases

def purchase_roc_plot(predictions, _array, xlab, ylab, title):
    probabilities, purchases = purchase_roc_curve(predictions, itemCntArray)
    # rank_cust = range(len(probabilities))
    plt.plot(probabilities, purchases)
    plt.plot([0,1], [0,1], ls="--")
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.show()

def roc_plot(fpr, tpr, xlab, ylab, title, label, outfile):
    model_data, = plt.plot(fpr, tpr, color="blue", label=label)
    random_data, = plt.plot([0,1], [0,1], ls="--", color='red', label="Random")
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.legend(handles=[model_data, random_data], loc=2)
    plt.savefig(outfile)
    plt.close()

def plot_roc(X, y, clf_class, outfile, **kwargs):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    kf = StratifiedKFold(y, n_folds=5, shuffle=True)
    y_prob = np.zeros((len(y),2))
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    for i, (train_index, test_index) in enumerate(kf):
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        # Predict probabilities, not classes
        y_prob[test_index] = clf.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y[test_index], y_prob[test_index, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    mean_tpr /= len(kf)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(outfile)
    plt.close()

############################################
#### CLV plots                          ####
####                                    ####
############################################

def plot_history_alive_min_thresholds(model, summary, transaction_data, threshold):

    """Plotting function for threshold plot of min customer alive probability
        vs pct cumulative customers with probability.
    Parameters
    ----------
    model : bgf model, fit to all data
    summary : summary set of data, output from summary_data_from_transaction_data function
    Returns
    -------
    saved figure
    """
    from lifetimes.utils import coalesce, calculate_alive_path
    #make a summary frame with customers with more than one purchase
    summary_multiple = summary[summary['frequency']>0]
    #find all purchase paths for customers, save in a list and then append to paths list
    paths = []
    for customer in summary_multiple.index:
        individual = summary.loc[[customer]]
        sp_trans = transaction_data.ix[transaction_data['CustomerNo'] == individual.index[0]]
        path = calculate_alive_path(model, sp_trans, 'OrderDate', int(individual['T']), '1D')
        paths.append(path)
    #find the min path prob for each individual
    min_paths = [path.min() for path in paths]

    #sort them, then plot the cumulative totals for each threshold (max = 0.7673)
    #make a threshold
    y = np.arange(0, len(min_paths))/len(min_paths)
    ax = plt.scatter(sorted(min_paths), y)
    plt.xlabel('min probability active')
    plt.ylabel('cumulative fraction of customers')
    plt.title('Fraction of customers with Min Probablity Active')
    return ax

############################################
#### NMF plots                          ####
####                                    ####
############################################

def plot_reconstruction_error(matrix, lower, upper):
    """Plotting function for reconstruction error of NMF models vs the number of components
    Parameters
    ----------
    matrix : pivoted input matrix for NMF fitting
    lower : lower bound on number of components
    upper : upper bound on number of components
    Returns
    -------
    saved figure
    """
    nmf_results = []
    for k in range(lower, upper+1):
        model = NMF(n_components=k, init='random', random_state=0)
        model.fit(matrix)
        nmf_results.append((k, model.reconstruction_err_))
    ax = plt.scatter(*zip(*nmf_results))
    plt.xlabel('N Clusters')
    plt.ylabel('Reconstruction Error')
    plt.title('N Clusters Vs Associated Reconstruction Error')
    return ax
