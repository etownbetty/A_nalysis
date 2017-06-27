import sys

#import data prep functions
from src.load_data import prep_data
from src.load_data import subset_data

from src.visualization import plot_history_alive_min_thresholds

import pandas as pd
import numpy as np

#import lifetimes modules
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter

from lifetimes.utils import summary_data_from_transaction_data
from lifetimes.utils import calibration_and_holdout_data

from lifetimes.plotting import plot_frequency_recency_matrix
from lifetimes.plotting import plot_period_transactions
from lifetimes.plotting import plot_calibration_purchases_vs_holdout_purchases
from lifetimes.plotting import plot_history_alive
from lifetimes.plotting import plot_probability_alive_matrix
from lifetimes.plotting import plot_expected_repeat_purchases

import matplotlib.pyplot as plt

class transactions(object):

    def summary_create(self, df):
        '''
        Subset df on sales data, create trans summary
        '''
        sales = subset_data(df, 'OrderType', 1)
        #make sure all sales kosher - keep only +0 sales
        sales = sales[sales.OrderTotal>0]
        self.transaction_data = sales[['OrderDate', 'CustomerNo']]
        return summary_data_from_transaction_data(self.transaction_data, 'CustomerNo', 'OrderDate', observation_period_end='2017-02-08')

    def fit_bgf(self, df, t):
        self.bgf = BetaGeoFitter()
        self.bgf.fit(df['frequency'], df['recency'], df['T'])
        self.viz_bgf(t)

    def viz_bgf(self, t):
        #visualize customer frequency and recency matrix
        plot_frequency_recency_matrix(self.bgf, T=t, cmap='coolwarm')
        plt.savefig('sales_frequency_recency_matrix.png')
        plt.close()
        #visualize customer alive probability
        plot_probability_alive_matrix(self.bgf, cmap='coolwarm')
        plt.savefig('probability_alive_matrix.png')
        plt.close()
        #visualize expected repeat Purchases
        plot_expected_repeat_purchases(self.bgf)
        plt.savefig('ProbabilityExpectedRepeatPurchases.png')
        plt.close()
        #visualize the expected number of period transactions
        plot_period_transactions(self.bgf)
        plt.savefig('period_transactions.png')
        plt.close()

    def predict_bgf_indiv(self, df, t, indiv):
        '''
        Predict transactions for a customer for a time frame (days)
        Save transaction visualization for the customer
        '''
        #predict purchases in t days
        df['predicted_purchases'] = df.apply(lambda r: self.bgf.conditional_expected_number_of_purchases_up_to_time(t, r['frequency'], r['recency'], r['T']), axis=1)
        print(df.sort_values('predicted_purchases').tail(5))
        #plot the customer history data with respect to being alive
        self.individual = df.loc[[indiv]]
        self.bgf.predict(t, self.individual['frequency'], self.individual['recency'], self.individual['T'])
        # print(self.bgf.summary())
        self.sp_trans = self.transaction_data.ix[self.transaction_data['CustomerNo'] == self.individual.index[0]]
        self.plot_history_alive_indiv(df, indiv)

    def plot_history_alive_indiv(self, df, indiv):
        '''
        Plot history alive/active for single customer
        '''
        plot_history_alive(self.bgf, int(self.individual['T']), self.sp_trans, 'OrderDate')
        plt.savefig('ProbabilityAliveByHistory_Customer{}.png'.format(indiv))
        plt.close()

    def plot_history_alive_all(self, df, threshold):
        '''
        Plot visualization to make a rule for marketing threshold
        '''
        plot_history_alive_min_thresholds(self.bgf, df, self.transaction_data, threshold)
        #put horizontal line on plot at threshold
        plt.savefig("CustomerThresholdsMinProbabilityActive.png")
        plt.close()

    def calibrate_bgf(self, calib_end_date, period_end_date, viz=False):
        '''
        Visualize the goodness of fit of BGF model
        '''
        summary_cal_holdout = calibration_and_holdout_data(self.transaction_data, 'CustomerNo', 'OrderDate',
                                            calibration_period_end=calib_end_date, #use 75% of data for training
                                            observation_period_end=period_end_date )
        if viz==True:
            print(summary_cal_holdout.head())

        self.bgf.fit(summary_cal_holdout['frequency_cal'], summary_cal_holdout['recency_cal'], summary_cal_holdout['T_cal'])
        plot_calibration_purchases_vs_holdout_purchases(self.bgf, summary_cal_holdout, colormap='coolwarm', alpha=0.75)
        plt.savefig('calibration_purchases_vs_holdout_purchases.png')
        plt.close()

class transactionMonetary(object):

    def summary_trans_create(self, df):
        '''
        Subset df on sales data, return trans summary with monetary spend
        '''
        sales = subset_data(df, 'OrderType', 1)
        sales = sales[sales.OrderTotal>0]
        transaction_data_monetary = sales[['OrderDate', 'CustomerNo', 'OrderTotal']]
        self.summary_monetary = summary_data_from_transaction_data(transaction_data_monetary, 'CustomerNo', 'OrderDate', 'OrderTotal', observation_period_end='2017-02-08')
        #keep customers with more than one spend
        self.return_customers = self.summary_monetary[self.summary_monetary['frequency']>0]
        return self.return_customers

    def fit_ggf(self):
        self.ggf = GammaGammaFitter(penalizer_coef = 0)
        self.ggf.fit(self.return_customers['frequency'], self.return_customers['monetary_value'])

    def summaryOutput(self, discount_rate=0.12, months=12):
        '''
        Fit beta geometric model to calculate CLV, and use GG model to calculate expected profit
        Per customer
        Write out CLV and profits to csv, print out averages to screen
        '''
        beta_model = BetaGeoFitter()
        #calulate average transaction value
        self.summary_monetary['avg_transaction_value'] = self.ggf.conditional_expected_average_profit(
        self.summary_monetary['frequency'],
        self.summary_monetary['monetary_value'])
        #fit beta geo model
        beta_model.fit(self.summary_monetary['frequency'], self.summary_monetary['recency'], self.summary_monetary['T'])
        #calculate clv, with discount rate calulated over year (default)
        disc_rate = discount_rate/months/30
        self.summary_monetary['clv'] = self.ggf.customer_lifetime_value(
        beta_model, #the model to use to predict the number of future transactions
        self.summary_monetary['frequency'],
        self.summary_monetary['recency'],
        self.summary_monetary['T'],
        self.summary_monetary['monetary_value'], time=months, # months
        discount_rate=disc_rate # monthly discount rate ~ 12.7% annually
        )
        #print customer data with calculations
        self.summary_monetary.to_csv("CLV_AVG_transactionValue_perCustomer.csv", index=False)
        #print summary stats
        print("Expected conditional average profit: {}, Average profit: {}".format(
        self.ggf.conditional_expected_average_profit(
            self.summary_monetary['frequency'],
            self.summary_monetary['monetary_value']).mean(),
        self.summary_monetary[self.summary_monetary['frequency']>0]['monetary_value'].mean()))

if __name__=="__main__":

    trans = transactions()
    order_file = sys.argv[1]
    #read in data with purachase dates and totals
    order = prep_data(order_file, ind_cols=['OrderType'], ind_pos=['Sales'], ind_newcols=['OrderType'])
    #make transaction data
    summary = trans.summary_create(order)
    #fit betageometric model and output visualizations with respect to 30 day time period
    trans.fit_bgf(summary, 30)
    #make a prediction on a customer and output customer purchase history
    trans.predict_bgf_indiv(summary, 30, 19563)
    #make visualization of all min customer alive points, with threshold point
    trans.plot_history_alive_all(summary, 0.1)

    #min date = 2011-08-17
    #max date = 2017-02-08
    #check the viability of the model, with "training" and "test" data
    trans.calibrate_bgf('2015-10-15', '2017-02-08', True)

    #do analysis with monetary spend, and gamma gamma model
    transM = transactionMonetary()
    #create customer summary and transaction data set, get returning customer data
    returning_customers_summary = transM.summary_trans_create(order)
    #make sure that GammaGamma assumptions are met - independence of spend and frequency of visits
    returning_customers_summary[['monetary_value', 'frequency']].corr()
    #fit a gamma model with the summary info, made in the class
    transM.fit_ggf()
    #print out customer summary data into csv
    transM.summaryOutput(discount_rate=0.12)
