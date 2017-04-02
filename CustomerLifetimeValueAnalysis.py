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

if __name__=="__main__":

    order_file = sys.argv[1]
    #read in data with purachase dates and totals
    order = prep_data(order_file, ind_cols=['OrderType'], ind_pos=['Sales'], ind_newcols=['OrderType'])
    #create sales dataframe
    sales = subset_data(order, 'OrderType', 1)
    #make sure all sales kosher - keep only +0 sales
    sales = sales[sales.OrderTotal>0]
    #make a transaction_data set
    transaction_data = sales[['OrderDate', 'CustomerNo']]
    summary = summary_data_from_transaction_data(transaction_data, 'CustomerNo', 'OrderDate', observation_period_end='2017-02-08')

    bgf = BetaGeoFitter()
    bgf.fit(summary['frequency'], summary['recency'], summary['T'])
    print(bgf)

    #visualize customer frequency and recency matrix
    plot_frequency_recency_matrix(bgf, T=30, cmap='coolwarm')
    plt.savefig('sales_frequency_recency_matrix.png')
    plt.close()

    #visualize customer alive probability
    plot_probability_alive_matrix(bgf, cmap='coolwarm')
    plt.savefig('probability_alive_matrix.png')
    plt.close()

    #visualize expected repeat Purchases
    plot_expected_repeat_purchases(bgf)
    plt.savefig('ProbabilityExpectedRepeatPurchases.png')
    plt.close()

    t = 30 #predict purchases in 30 days
    summary['predicted_purchases'] = summary.apply(lambda r: bgf.conditional_expected_number_of_purchases_up_to_time(30, r['frequency'], r['recency'], r['T']), axis=1)
    summary.sort_values('predicted_purchases').tail(5)

    #visualize the expected number of period transactions
    plot_period_transactions(bgf)
    plt.savefig('period_transactions.png')
    plt.close()

    #plot the customer history data with respect to being alive
    individual = summary.loc[[19563]]
    bgf.predict(t, individual['frequency'], individual['recency'], individual['T'])

    sp_trans = transaction_data.ix[transaction_data['CustomerNo'] == individual.index[0]]
    plot_history_alive(bgf, int(individual['T']), sp_trans, 'OrderDate')
    plt.savefig('ProbabilityAliveByHistory.png')
    plt.close()

    #visualize to make a rule for marketing threshold
    plot_history_alive_min_thresholds(bgf, summary)
    plt.savefig("CustomerThresholdsMinProbabilityAlive.png")
    plt.close()

    #min date = 2011-08-17
    #max date = 2017-02-08
    #check the viability of the model, with "training" and "test" data
    summary_cal_holdout = calibration_and_holdout_data(transaction_data, 'CustomerNo', 'OrderDate',
                                        calibration_period_end='2015-10-15', #use 75% of data for training
                                        observation_period_end='2017-02-08' )
    print(summary_cal_holdout.head())

    bgf.fit(summary_cal_holdout['frequency_cal'], summary_cal_holdout['recency_cal'], summary_cal_holdout['T_cal'])
    plot_calibration_purchases_vs_holdout_purchases(bgf, summary_cal_holdout, colormap='coolwarm', alpha=0.75)
    plt.savefig('calibration_purchases_vs_holdout_purchases.png')
    plt.close()

    #do analysis with monetary spend
    transaction_data_monetary = sales[['OrderDate', 'CustomerNo', 'OrderTotal']]
    summary_monetary = summary_data_from_transaction_data(transaction_data_monetary, 'CustomerNo', 'OrderDate', 'OrderTotal', observation_period_end='2017-02-08')

    returning_customers_summary = summary_monetary[summary_monetary['frequency']>0]
    #make sure that GammaGamma assumptions are met - independence of spend and frequency of visits
    returning_customers_summary[['monetary_value', 'frequency']].corr()

    ggf = GammaGammaFitter(penalizer_coef = 0)
    ggf.fit(returning_customers_summary['frequency'], returning_customers_summary['monetary_value'])

    #heavily based on previous spending totals
    print(ggf.conditional_expected_average_profit(
        summary_monetary['frequency'],
        summary_monetary['monetary_value']).sort_values(ascending=False).head())

    print("Expected conditional average profit: {}, Average profit: {}".format(
    ggf.conditional_expected_average_profit(
        summary_monetary['frequency'],
        summary_monetary['monetary_value']).mean(),
    summary_monetary[summary_monetary['frequency']>0]['monetary_value'].mean()))

    bgf.fit(summary_monetary['frequency'], summary_monetary['recency'], summary_monetary['T'])

    # #this is calculated with a discount rate??
    # print(ggf.customer_lifetime_value(
    # bgf, #the model to use to predict the number of future transactions
    # summary_monetary['frequency'],
    # summary_monetary['recency'],
    # summary_monetary['T'],
    # summary_monetary['monetary_value'], time=12, # months
    # discount_rate=0.01 # monthly discount rate ~ 12.7% annually
    # ).head(10))
