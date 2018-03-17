# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

def dealWithUnixTime(df,col='context_timestamp'):
    df[col] = df[col].astype(int)
    df[col] = pd.to_datetime(df[col],unit='s')
    df = df[df[col] > '1970-01-01']
    df = df.sort_index(by=col)
    df['day'] = [i.day for i in df[col]]
    df['hour'] = [i.hour for i in df[col]]
    df['minute'] = [i.minute for i in df[col]]
    df['second'] = [i.second for i in df[col]]
    return df

def changeType(df):
    df['instance_id'] = df['instance_id'].astype('int64')
    df['item_id'] = df['item_id'].astype('int64')
    df['item_brand_id'] = df['item_brand_id'].astype('int64')
    df['item_city_id'] = df['item_city_id'].astype('int64')
    df['user_id'] = df['user_id'].astype('int64')
    df['user_age_level'] = dealWithNan(df['user_age_level'],1000)
    df['user_occupation_id'] = dealWithNan(df['user_occupation_id'],2000)
    df['user_star_level'] = dealWithNan(df['user_star_level'],3000)
    df['context_id'] = df['context_id'].astype('int64')
    df['context_page_id'] = dealWithNan(df['context_page_id'],4000)
    df['shop_id'] = df['shop_id'].astype('int64')
    df['shop_star_level'] = dealWithNan(df['shop_star_level'],4999)
    return df

def dealWithNan(SR,level):
    l = []
    for sr in SR:
        if sr == -1:
            l.append(sr)
        else :
            l.append(sr - level)
    return l

def dropNan(data):
    data_nan = (data == -1)
    total = data_nan.sum().sort_values(ascending=False)
    percent = (data_nan.sum()/data_nan.count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    missing_data =  missing_data[missing_data.Total > 0]
    print(missing_data)
    data = data[data != -1] 
    data = data.dropna() #将nan值drop掉
    del data_nan
    return data

def doDummy(data,l_feature):
    for col in l_feature:
        df_temp = pd.get_dummies(data[col],prefix=col,drop_first=True)
        data = data.drop(col,axis=1)
        data = pd.concat([data,df_temp],axis=1)
    del df_temp
    return data

def splitData(data):
    train = data[data.context_timestamp <= '2018-09-23']
    test = data[data.context_timestamp > '2018-09-23']
    return train,test



if __name__ == '__main__':

    data = pd.read_csv('../raw_data/round1_train.csv')
    data['is_trade'].describe()
    
    
    data = dealWithUnixTime(data)
    l = list(data['context_timestamp'])
    print('样本时间窗：')
    print('%s - %s' %(l[0].strftime("%Y-%m-%d"),l[-1].strftime("%Y-%m-%d")))
    del l
    
    data = changeType(data)
    data = dropNan(data)
    
    l_dummy_feature = ['item_price_level','item_sales_level','item_collected_level','item_pv_level',
                       'user_gender_id','user_age_level','user_occupation_id','user_star_level',
                       'context_page_id','shop_review_num_level','shop_star_level','day','hour']
    
    data = doDummy(data,l_dummy_feature)
    df_train, df_test = splitData(data)
    
    params = {
                'objective': 'binary:logistic',
                'eta': 0.1,
                'max_depth': 7,
                'eval_metric': 'auc',
                'subsample' : 0.8,
                'colsample_bytree' : 0.8,
                'silent' : 0,
                'seed' : 8,
                'scale_pos_weight' : 40 
            }
    
    feature=[x for x in df_train.columns if x not in ['is_trade','item_category_list','item_property_list',
                                                      'context_timestamp','predict_category_property']]
    xgbtrain = xgb.DMatrix(df_train[feature],df_train['is_trade'])
    xgbtest = xgb.DMatrix(df_test[feature],df_test['is_trade'])
    watchlist = [ (xgbtrain,'train'), (xgbtest, 'test') ]
    num_rounds=5000
    print('Start training...')
    model = xgb.train(params, xgbtrain, num_rounds, watchlist, early_stopping_rounds=25)


