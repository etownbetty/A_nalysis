import numpy as np
import pandas as pd

def prep_data(filename, ind_cols=None, ind_pos=None, ind_newcols=None, dt_cols=None):
    ## takes in a data path, with specifications for columns that should be made into indicator variables
    ## returns a pandas data frame, with new int variables in place of objects
    if ind_cols:
        df = pd.read_csv(filename, parse_dates=dt_cols, date_parser = pd.tseries.tools.to_datetime)
        for i in range(len(ind_cols)):
            df[ind_newcols[i]] = np.where(df[ind_cols[i]]==ind_pos[i], 1, 0)
    else:
        df = pd.read_csv(filename)
    return df

def subset_data(df, col, subset_on):
    ##takes in df, column to partition on, value to partition on
    ##returns subset data in pandas dataframe
    return df[df[col]==subset_on]

def get_first_sale_data(df, N):
    max_date = df['OrderDate'].max()
    df_N = df[df['number_sales']==N]
    df_N_first = df_N[['CustomerNo', 'OrderDate', 'OrderTotal', 'ItemCnt']].sort_values(['CustomerNo','OrderDate']).groupby('CustomerNo').first()
    #change timedelta variable to int
    df_N_first['loyalty'] = ((max_date-df_N_first['OrderDate'])/np.timedelta64(1, 'D')).astype(int)
    #rename variables
    df_return = df_N_first.rename(columns={'OrderTotal':'FirstOrderTotal', 'ItemCnt':'FirstItemCnt'})
    return df_return

def get_last_sale_data(df, N):
    df_N = df[df['number_sales']==N]
    df_N_0 = df_N.dropna()
    df_N_0_last = df_N_0.sort_values(['CustomerNo','OrderDate']).groupby('CustomerNo').last()
    df_N_0_last['CustomerNo'] = df_N_0_last.index
    #change timedelta variable to int
    df_N_0_last['time_since_purchase'] = (df_N_0_last['diff']/np.timedelta64(1, 'D')).astype(int)
    return df_N_0_last

def partition_sale_data(df, N, gt_than=False):

    #read in pandas dataframe with aggregated transactions per customer and other customer data
    #returns pandas subset dataframe with only customers who had "N" number of transactions
    # or customers who had greater than "N" number of transactions and time since last purchase

    #create time diff variable
    df['diff'] = df.sort_values(['CustomerNo','OrderDate']).groupby('CustomerNo')['OrderDate'].diff()

    #for customers who have more than one sale,
    if N>1:
        if gt_than==False:
            df_N_0_last = get_last_sale_data(df, N)
            #for those customers who have two purchases on 1 day, fill in 1 day in between
            # df_N_0_last['time_since_purchase'][df_N_0_last['time_since_purchase'] == 0] = 1
            #attach first purchase info, amount and item count, and "loyalty"
            df_N_first = get_first_sale_data(df, N)
            df_return = df_N_0_last.merge(df_N_first, left_on='CustomerNo', right_index=True)
        else:
            df_N_0_last = get_last_sale_data(df, N)
            #for those customers who have two purchases on 1 day, fill in 1 day in between
            # df_N_0_last['time_since_purchase'][df_N_0_last['time_since_purchase'] == 0] = 1
            df_N_first = get_first_sale_data(df, N)
            df_return = df_N_0_last.merge(df_N_first, left_on='CustomerNo', right_index=True)
        return df_return.drop(['OrderNo', 'diff', 'OrderDate_x', 'OrderDate_y', 'OrderType'], axis=1).rename(columns={'OrderTotal':'LastOrderTotal', 'ItemCnt':'LastItemCnt'})
    else:
        #for customers who only have one sale, keep the zero diff
        max_date = df['OrderDate'].max()
        df_N = df[df['number_sales']==1]
        df_N_0 = df_N.fillna(0)
        df_N_0['loyalty'] = ((max_date-df_N_0['OrderDate'])/np.timedelta64(1, 'D')).astype(int)
        return df_N_0.drop(['OrderType', 'number_sales', 'OrderDate', 'OrderNo', 'diff'], axis=1)
