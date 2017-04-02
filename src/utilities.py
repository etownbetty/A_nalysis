import numpy as np

from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

class modeModel():
    def __init__(self):
        self.y = None
    def fit(self, y):
        self.y = stats.mode(y).mode[0]
    def predict(self, X):
        return np.repeat(self.y, len(X))


def div_count_pos_neg(X, y):
    """Helper function to divide X & y into positive and negative classes
    and counts up the number in each.

    Parameters
    ----------
    X : ndarray - 2D
    y : ndarray - 1D

    Returns
    -------
    negative_count : Int
    positive_count : Int
    X_positives    : ndarray - 2D
    X_negatives    : ndarray - 2D
    y_positives    : ndarray - 1D
    y_negatives    : ndarray - 1D
    """
    negatives, positives = y == 0, y == 1
    negative_count, positive_count = np.sum(negatives), np.sum(positives)
    X_positives, y_positives = X[positives], y[positives]
    X_negatives, y_negatives = X[negatives], y[negatives]
    return negative_count, positive_count, X_positives, \
           X_negatives, y_positives, y_negatives



def undersample(X, y, tp):
    """Randomly discards negative observations from X & y to achieve the
    target proportion of positive to negative observations.

    Parameters
    ----------
    X  : ndarray - 2D
    y  : ndarray - 1D
    tp : float - range [0.5, 1], target proportion of positive class observations

    Returns
    -------
    X_undersampled : ndarray - 2D
    y_undersampled : ndarray - 1D
    """
    if tp < np.mean(y):
        return X, y
    neg_count, pos_count, X_pos, X_neg, y_pos, y_neg = div_count_pos_neg(X, y)
    negative_sample_rate = (pos_count * (1 - tp)) / (neg_count * tp)
    negative_keepers = np.random.choice(a=[False, True], size=neg_count,
                                        p=[1 - negative_sample_rate,
                                           negative_sample_rate])
    X_negative_undersampled = X_neg[negative_keepers]
    y_negative_undersampled = y_neg[negative_keepers]
    X_undersampled = np.vstack((X_negative_undersampled, X_pos))
    y_undersampled = np.concatenate((y_negative_undersampled, y_pos))

    return X_undersampled, y_undersampled

def oversample(X, y, tp):
    """Randomly choose positive observations from X & y, with replacement
    to achieve the target proportion of positive to negative observations.

    Parameters
    ----------
    X  : ndarray - 2D
    y  : ndarray - 1D
    tp : float - range [0, 1], target proportion of positive class observations

    Returns
    -------
    X_undersampled : ndarray - 2D
    y_undersampled : ndarray - 1D
    """
    if tp < np.mean(y):
        return X, y
    neg_count, pos_count, X_pos, X_neg, y_pos, y_neg = div_count_pos_neg(X, y)
    positive_range = np.arange(pos_count)
    positive_size = (tp * neg_count) / (1 - tp)
    positive_idxs = np.random.choice(a=positive_range,
                                     size=int(positive_size),
                                     replace=True)
    X_positive_oversampled = X_pos[positive_idxs]
    y_positive_oversampled = y_pos[positive_idxs]
    X_oversampled = np.vstack((X_positive_oversampled, X_neg))
    y_oversampled = np.concatenate((y_positive_oversampled, y_neg))

    return X_oversampled, y_oversampled

def smote(X, y, tp, k=None):
    """Generates new observations from the positive (minority) class.
    For details, see: https://www.jair.org/media/953/live-953-2037-jair.pdf

    Notes: Currently the KNeighborsClassifier will throw a warning when calling
           to the kneighbors method. Appears to be happening in sklearn not
           this usage of it.

    Parameters
    ----------
    X  : ndarray - 2D
    y  : ndarray - 1D
    tp : float - [0, 1], target proportion of positive class observations

    Returns
    -------
    X_smoted : ndarray - 2D
    y_smoted : ndarray - 1D
    """
    if tp < np.mean(y):
        return X, y
    if k is None:
        k = int(len(X) ** 0.5)

    neg_count, pos_count, X_pos, X_neg, y_pos, y_neg = div_count_pos_neg(X, y)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_pos, y_pos)
    neighbors = knn.kneighbors(return_distance=False)

    positive_size = (tp * neg_count) / (1 - tp)
    smote_num = int(positive_size - pos_count)

    rand_idxs = np.random.randint(0, pos_count, size=smote_num)
    rand_nghb_idxs = np.random.randint(0, k, size=smote_num)
    rand_pcts = np.random.random((smote_num, X.shape[1]))
    smotes = []
    for r_idx, r_nghb_idx, r_pct in zip(rand_idxs, rand_nghb_idxs, rand_pcts):
        rand_pos, rand_pos_neighbors = X_pos[r_idx], neighbors[r_idx]
        rand_pos_neighbor = X_pos[rand_pos_neighbors[r_nghb_idx]]
        rand_dir = rand_pos_neighbor - rand_pos
        rand_change = rand_dir * r_pct
        smoted_point = rand_pos + rand_change
        smotes.append(smoted_point)

    X_smoted = np.vstack((X, np.array(smotes)))
    y_smoted = np.concatenate((y, np.ones((smote_num,))))
    return X_smoted, y_smoted

#get scores from different classifier methods
def get_scores(classifier, X_train, X_test, y_train, y_test, **kwargs):
    model = classifier(**kwargs)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    return model.score(X_test, y_test), \
           precision_score(y_test, y_predict), \
           recall_score(y_test, y_predict)
