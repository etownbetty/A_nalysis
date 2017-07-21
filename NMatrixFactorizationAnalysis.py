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

class NMfactorization(object):

    def norm_pivot_data(self, items_df, order_df):
        '''
        Takes in data sets and returns normalized pivot and repeat purchase table for factorization
        '''
        items_sale = subset_data(items_df, 'OrderType', 'Sales')
        order_sale = subset_data(order_df, 'OrderType', 'Sales')
        #for each customer, the number of each item they purchased
        df = order_sale[['CustomerNo', 'OrderNo']].merge(items_sale[['OrderNo', 'ProductNo']], on='OrderNo', how='right')
        df_pivot = pd.pivot_table(df[[id_col, item_col]], index='CustomerNo', columns='ProductNo', aggfunc=len, fill_value=0)
        df_matrix = df_pivot.applymap(lambda x: x*1.)
        ##limit clustering to customers who bought more than one thing over lifetime
        self.df_norm = self.normalize_pivoted_table(df_matrix)
        self.df_repeat = df_matrix[df_matrix.sum(axis=1) > 1]
        self.norm_df_repeat = self.normalize_pivoted_table(self.df_repeat)
        return self.df_norm, self.df_norm_df_repeat

    def normalize_pivoted_table(self, df):
        row_sums = df.sum(axis=1)
        return df.as_matrix()/ row_sums[:, np.newaxis]

    def plot_reconstruction_error(matrix, lower, upper):
        '''
        Plotting function for reconstruction error of NMF models vs the number of components
        Parameters
        ----------
        matrix : pivoted input matrix for NMF fitting
        lower : lower bound on number of components
        upper : upper bound on number of components
        Returns
        -------
        saved figure
        '''
        nmf_results = []
        for k in range(lower, upper+1):
            model = NMF(n_components=k, init='random', random_state=0)
            model.fit(matrix)
            nmf_results.append((k, model.reconstruction_err_))
        ax = plt.scatter(*zip(*nmf_results))
        plt.xlabel('N Clusters')
        plt.ylabel('Reconstruction Error')
        plt.title('N Clusters Vs Associated Reconstruction Error')
        plt.savefig('NMF_model_reconstruction_error_range5_20.png')
        plt.close()

    def fit_nmf_model(self, mat):
        '''
        Does a non-negative matrix factorization on input matrix
        '''
        model = NMF(n_components=10, init='random', random_state=0)
        W = model.fit_transform(mat)
        H = model.components_
        return W,H

    def other_purchases(self, sku_df, item_id, N_items):
        '''
        Returns a list of N items also purchased with an item
        '''
        from collections import Counter
        #subset the rows that had the item_id
        cols = self.df_repeat.columns
        purchasers = self.df_repeat[self.df_repeat[item_id]>0]
        purchases = purchasers.apply(lambda x: x > 0)
        purchase_lists = purchases.apply(lambda x: list(cols[x.values]), axis=1)
        #now enumerate each list of purchases and put into dictionary
        all_purchases = [item for sublist in purchase_lists for item in sublist]
        common_items = [x[0] for x in Counter(all_purchases).most_common(N_items+1)]
        purchased_other_items = list(sku_df[sku_df['ProductNo'].isin(common_items)]['ProductName'].unique())
        purchases = "Actual Other Purchases with {}:".format(item_id)
        for i in range(10):
            purchases = ", ".join([purchases, purchased_other_items[i]])
        print(purchases)

    def return_topic_components(self, H, sku_df, topicN, n_components):
        '''
        Returns n items from the chosen topicN
        '''
        topic_num = topicN+1
        topic = zip(H[topic_num], self.df_repeat.columns.values)
        topic_sorted = sorted(topic, key=lambda x: x[0], reverse=True)
        components = []
        for i in range(n_components):
            components.append(sku_df[sku_df['ProductNo']==topic_sorted[:10][i][1]].iloc[0]['ProductNo'])
        return components

    def print_topic_components(self, H, sku_df, topicN, n_components):
        '''
        Prints out top n items from topicN
        '''
        topic_num = topicN+1
        topic = zip(H[topic_num], self.df_repeat.columns.values)
        topic_sorted = sorted(topic, key=lambda x: x[0], reverse=True)
        components = "Topic {}:".format(topicN)
        for i in range(n_components):
            components = ", ".join([components, str(sku_df[sku_df['ProductNo']==topic_sorted[:10][i][1]].iloc[0]['ProductName'])])
        return components

    def print_n_topic_components(self, N, H, sku_df, n_components):
        '''
        Prints out top n items from N topics
        '''
        for i in range(N):
            topicN = i+1
            print(print_topic_components(H, sku_df, topicN, n_components), "\n")

    def fit_mat(self, mat):
        '''
        Return item similarity matrix for repeat customers
        '''
        item_sim_mat = cosine_similarity(mat.T)
        return item_sim_mat

    def _set_neighborhoods(self, item_sim_mat, neighborhood_size):
        '''
        Returns neighborhoods of items, of neighborhood_size
        '''
        least_to_most_sim_indexes = np.argsort(item_sim_mat, 1)
        neighborhoods = least_to_most_sim_indexes[:, neighborhood_size:]
        return neighborhoods

    def pred_one_user(self, user_id, mat):
        '''
        Prints out a single item prediction for a user
        '''
        item_sim_mat = cosine_similarity(mat.T)
        neighborhoods = self._set_neighborhoods(item_sim_mat, 100)
        #get the indexes of the items
        items_purchased_by_this_user = self.df_repeat[self.df_repeat.index==user_id].as_matrix().nonzero()[1]
        # Just initializing so we have somewhere to put rating preds
        out = np.zeros(self.df_repeat.shape[1])
        for item_to_rate in range(self.df_repeat.shape[1]):
            relevant_items = np.intersect1d(neighborhoods[item_to_rate],
                                            items_purchased_by_this_user,
                                            assume_unique=True)  # assume_unique speeds up intersection op
            out[item_to_rate] = self.df_repeat[user_id, [relevant_items]] * \
                item_sim_mat[item_to_rate, relevant_items] / \
                item_sim_mat[item_to_rate, relevant_items].sum()
        cleaned_out = np.nan_to_num(out)
        return cleaned_out

    def top_n_recs(self, user_id, n, purch_mat):
        '''
        Top N recommendations for a user id
        '''
        pred_ratings = self.pred_one_user(user_id)
        item_index_sorted_by_pred_rating = list(np.argsort(pred_ratings))
        items_rated_by_this_user = purch_mat[user_id].nonzero()[1]
        unrated_items_by_pred_rating = [item for item in item_index_sorted_by_pred_rating
                                        if item not in items_rated_by_this_user]
        return unrated_items_by_pred_rating[-n:]

if __name__ == "__main__":

    #load in all data
    sku_df = prep_data(sys.argv[1])
    items_df = prep_data(sys.argv[2])
    order_df = prep_data(sys.argv[3])

    nm = NMfactorization()
    #return pivoted matrix, and repeat purchase pivoted matrix
    mat, repeat_mat = nm.norm_pivot_data(items_df, order_df)
    #fit NMF model with 10 components
    W,H = nm.fit_nmf_model(mat)

    nm.plot_reconstruction_error(mat, 5, 20)

    print("Items from the top 3 topics: \n")
    nm.print_n_topic_components(3, H, sku_df, 10)

    #fit NMF model with 10 components for repeat purchasers
    W_repeat,H_repeat = nm.fit_nmf_model(repeat_mat)

    print("Items from the top 2 topics: \n")
    nm.print_n_topic_components(2, H_repeat, sku_df, 10)

    #10 items in topic 1
    topic1 = nm.return_topic_components(H_repeat, sku_df, 1, 10)

    #checking the topic clusters for overlap with actual Purchases, aka Sanity Check of clusters
    actual_other_purchases = nm.other_purchases(sku_df, topic1[0], 10)
    print("Other items bought with top item in topic #1, {}".format(actual_other_purchases))
    #for a repeat purchaser, find a purchase from the topics and recommend an additional purchase
    pred_15198 = nm.pred_one_user(15198, repeat_mat)
