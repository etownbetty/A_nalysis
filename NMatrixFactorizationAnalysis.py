import sys

#import data prep functions
from src.load_data import prep_data
from src.load_data import subset_data
from src.load_data import pivot_data

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import scipy

#model imports
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity

#plotting
import matplotlib.pyplot as plt

def normalize_pivoted_table(df):
    row_sums = df.sum(axis=1)
    return df.as_matrix()/ row_sums[:, np.newaxis]

def fit_nmf_model(mat):
    model = NMF(n_components=10, init='random', random_state=0)
    W = model.fit_transform(mat)
    H = model.components_
    return W,H

def other_purchases(pivot_frame, sku_df, item_id, N_items, id_col, name_col):
    from collections import Counter
    #subset the rows that had the item_id
    cols = pivot_frame.columns
    purchasers = pivot_frame[pivot_frame[item_id]>0]
    purchases = purchasers.apply(lambda x: x > 0)
    purchase_lists = purchases.apply(lambda x: list(cols[x.values]), axis=1)
    #now enumerate each list of purchases and put into dictionary
    all_purchases = [item for sublist in purchase_lists for item in sublist]
    common_items = [x[0] for x in Counter(all_purchases).most_common(N_items+1)]
    purchased_other_items = list(sku_df[sku_df[id_col].isin(common_items)][name_col].unique())
    purchases = "Actual Other Purchases with {}:".format(item_id)
    for i in range(10):
        purchases = ", ".join([purchases, purchased_other_items[i]])
    print(purchases)

def return_topic_components(H, df, sku_df, topicN, n_components, id_col, name_col):
    topic_num = topicN+1
    topic = zip(H[topic_num], df.columns.values)
    topic_sorted = sorted(topic, key=lambda x: x[0], reverse=True)
    components = []
    for i in range(n_components):
        components.append(sku_df[sku_df[id_col]==topic_sorted[:10][i][1]].iloc[0][name_col])
    return components

def print_topic_components(H, df, sku_df, topicN, n_components, id_col, name_col):
    topic_num = topicN+1
    topic = zip(H[topic_num], df.columns.values)
    topic_sorted = sorted(topic, key=lambda x: x[0], reverse=True)
    components = "Topic {}:".format(topicN)
    for i in range(n_components):
        components = ", ".join([components, str(sku_df[sku_df[id_col]==topic_sorted[:10][i][1]].iloc[0][name_col])])
    return components

def print_n_topic_components(N, H, df, sku_df, topicN, n_components, id_col, name_col):
    for i in range(N):
        topicN = i+1
        print(print_topic_components(H, df, sku_df, topicN, n_components, id_col, name_col), "\n")

def fit_mat(purch_mat):
    n_users = purch_mat.shape[0]
    n_items = purch_mat.shape[1]
    item_sim_mat = cosine_similarity(purch_mat.T)
    return item_sim_mat

def _set_neighborhoods(item_sim_mat, neighborhood_size):
    least_to_most_sim_indexes = np.argsort(item_sim_mat, 1)
    neighborhoods = least_to_most_sim_indexes[:, neighborhood_size:]
    return neighborhoods

def pred_one_user(user_id, purch_mat, neighborhoods, fit_mat):
    #get the indexes of the items
    items_purchased_by_this_user = purch_mat[purch_mat.index==user_id].as_matrix().nonzero()[1]
    # Just initializing so we have somewhere to put rating preds
    out = np.zeros(purchased.shape[1])
    for item_to_rate in range(purch_mat.shape[1]):
        relevant_items = np.intersect1d(neighborhoods[item_to_rate],
                                        items_purchased_by_this_user,
                                        assume_unique=True)  # assume_unique speeds up intersection op
        out[item_to_rate] = purch_mat[user_id, [relevant_items]] * \
            fit_mat[item_to_rate, relevant_items] / \
            fit_mat[item_to_rate, relevant_items].sum()
    cleaned_out = np.nan_to_num(out)
    return cleaned_out

def top_n_recs(user_id, n, pred_one_user, purch_mat):
    pred_ratings = pred_one_user(user_id)
    item_index_sorted_by_pred_rating = list(np.argsort(pred_ratings))
    items_rated_by_this_user = purch_mat[user_id].nonzero()[1]
    unrated_items_by_pred_rating = [item for item in item_index_sorted_by_pred_rating
                                    if item not in items_rated_by_this_user]
    return unrated_items_by_pred_rating[-n:]

if __name__ == "__main__":

    #load in the sku data
    sku_df = prep_data(sys.argv[1])

    #load in items in order data, subset for sales
    items_df = prep_data(sys.argv[2])
    items_sale = subset_data(items_df, 'OrderType', 'Sales')

    #load in order data, subset for sales
    order_df = prep_data(sys.argv[3])
    order_sale = subset_data(order_df, 'OrderType', 'Sales')

    #for each customer, the number of each item they purchased
    df = order_sale[['CustomerNo', 'OrderNo']].merge(items_sale[['OrderNo', 'ProductNo']], on='OrderNo', how='right')
    #pivot data with column header data
    df_matrix = pivot_data(df, 'CustomerNo', 'ProductNo')
    #normalize data
    new_matrix = normalize_pivoted_table(df_matrix)

    #fit NMF model with 10 components, seems reasonable to have about 10 different topics of purchases
    W,H = fit_nmf_model(new_matrix)

    plot_reconstruction_error(new_matrix, 5, 20)
    plt.savefig('NMF_model_reconstruction_error_range5_20.png')
    plt.close()

    print_n_topic_components(3, H_cut, df_2purch, sku_df, 1, 10, 'ProductNo', 'ProductName')

    ##limit clustering to customers who bought more than one thing over lifetime
    df_2purch = df_matrix[df_matrix.sum(axis=1) > 1]
    #normalize
    new_2purch = normalize_pivoted_table(df_2purch)

    W_cut,H_cut = fit_nmf_model(new_2purch)

    #print out top 3 topics
    print("Items from the top 2 topics: \n")
    print_n_topic_components(2, H_cut, df_2purch, sku_df, 1, 10, 'ProductNo', 'ProductName')

    topic1 = return_topic_components(H_cut, df_2purch, sku_df, 1, 10, 'ProductNo', 'ProductNo')

    #checking the topic clusters for overlap with actual Purchases, aka Sanity Check of clusters
    actual_other_purchases = other_purchases(df_2purch, sku_df, topic1[0], 10, 'ProductNo', 'ProductName')
    print("Other items bought with top item in topic #1, {}".format(actual_other_purchases))
    #for a user, find a purchase from the topics and recommend an additional purchase

    item_sim_mat = fit_mat(new_2purch)
    neighborhoods = _set_neighborhoods(item_sim_mat, 100)
    pred_15198 = pred_one_user(15198, df_2purch, neighborhoods, item_sim_mat)
