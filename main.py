# coding=utf-8
#!/usr/bin/python3
'''
Created on 2019年11月26日

@author: liushouhua
'''
import pandas as pd
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option("display.max_colwidth",100)
pd.set_option('display.width',1000)

import os
import sys

import xlrd
from sklearn.metrics import auc, roc_curve
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot as plt 
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
data = "data/"
def get_leaf(x_train, y_train, x_val):
    
    x_train, y_train, x_val = map(np.array, [x_train, y_train, x_val])
    x_train = x_train.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)
    x_val = x_val.reshape(-1, 1)
    m = DecisionTreeClassifier(min_samples_leaf=0.001, max_leaf_nodes=10)
    m.fit(x_train, y_train)
    return m.apply(x_val), roc_auc_score(y_train, m.predict_proba(x_train)[:, 1])

def business(buss):
    buss = buss[1:-1].split(",")
    return len(buss)

def data_review():
    
#     data_deal(istrain=True)
    df_train = pd.read_csv(data + "train_deal.csv", encoding='utf-8',low_memory=False)
#     data_deal(istrain=False)
    df_test = pd.read_csv(data + "test_deal.csv", encoding='utf-8',low_memory=False)
    
    
    features = ['企业类型',  '登记机关', '行业门类', '管辖机关']
    target_name= "Label"
    id_name = "ID"
    
    df_dt = pd.concat([df_train, df_test], sort=False, ignore_index=True).fillna(-1)
    for feature in features:
        df_dt[feature] = LabelEncoder().fit_transform(df_dt[feature].astype(np.str))
        df_dt[feature], auc = get_leaf(df_dt[feature].iloc[:-len(df_test)], df_dt[target_name].iloc[:-len(df_test)], df_dt[feature])
        count_table = pd.pivot_table(df_dt,
                                     index=feature,
                                     columns=target_name,
                                     values=id_name,
                                     aggfunc="count",
                                     fill_value=0)
        
        count_table["total"] = count_table[0] + count_table[1]
        count_table["rate"] = count_table[1] / np.sum(count_table[1]) - count_table[0] / np.sum(count_table[0])
        count_table = count_table.sort_values(by="rate")
        count_table["label"] = range(len(count_table))
        df_dt[feature] = df_dt[feature].replace(count_table.index, count_table.label)
        
    for feature in features:
        count_table = pd.pivot_table(df_dt,
                                     index=feature,
                                     columns=target_name,
                                     values=id_name,
                                     aggfunc="count",
                                     fill_value=0)
        count_table[[1, 0]] = count_table[[1, 0]] + 1
        count_table["rate_pos"] = count_table[1] / np.sum(count_table[1]) * 100
        count_table["rate_neg"] = count_table[0] / np.sum(count_table[0]) * 100
        count_table["efficiency"] = count_table["rate_pos"] - count_table["rate_neg"]
        count_table["rate"] = count_table[1] / (count_table[1] + count_table[0])
        count_table["woe"] = np.log(count_table["rate_pos"] / count_table["rate_neg"])
        count_table["iv"] = count_table["woe"] * count_table["efficiency"]
        count_table.drop([-1, 0, 1, "rate_pos", "rate_neg"], axis=1, inplace=True)
        count_table.columns = [feature + i for i in count_table.columns]
        df_dt = df_dt.merge(count_table.reset_index(), left_on=feature, right_on=feature, how="left").drop(feature, axis=1)

    train_d = df_dt.iloc[:-len(df_test)]
    test_d = df_dt.iloc[-len(df_test):]
    return train_d,test_d


def knn_miss(data_train,data_test,miss_clo):
    
    print("缺失值填充: ",miss_clo)
    df = pd.concat([data_train,data_test])
    scaler_co = ["注册资本","增值税","企业所得税","印花税","教育费","城建税"]
    df = df[["ID"]+scaler_co]
    scaler_co.remove(miss_clo)
    
    scaler = preprocessing.scale(df[scaler_co])
    df[scaler_co] = scaler
    
    train_data = df[df[miss_clo].notnull()].copy()
    test_data = df[df[miss_clo].isnull()].copy()
    if test_data.shape[0] == 0:
        return data_train,data_test

    for c in df.columns:
        train_data = train_data[train_data[c].notnull()]
    
    train_y = train_data[miss_clo]
    train_x = train_data.drop(["ID",miss_clo],axis=1)
    
    test_y = test_data[["ID",miss_clo]].copy()
    test_x = test_data.drop(["ID",miss_clo],axis=1)
    test_x.fillna(test_x.median(),inplace=True)
    
#     X_train, X_vaild, y_train, y_vaild = train_test_split(train_x, train_y, test_size=0.2, random_state=1000)
    knn_r = KNeighborsRegressor(n_neighbors=35, weights="uniform")
    knn_r.fit(train_x,train_y)
    y_pred = knn_r.predict(test_x)

    test_y[miss_clo] = y_pred
    
    ids = data_train[data_train[miss_clo].isnull()]["ID"].values
    data_train.loc[data_train["ID"].isin(ids),miss_clo] = test_y[test_y["ID"].isin(ids)][miss_clo]

    ids = data_test[data_test[miss_clo].isnull()]["ID"].values
    data_test.loc[data_test["ID"].isin(ids),miss_clo] = test_y[test_y["ID"].isin(ids)][miss_clo]
        
    return data_train,data_test

def data_deal(istrain=True):
    #["印花税","注册资本","增值税","企业所得税","教育费","城建税"]
#     istrain=False
    df_train = pd.read_csv(data + "train.csv", encoding='utf-8',low_memory=False)
    df_test = pd.read_csv(data + "test.csv", encoding='utf-8',low_memory=False)
    columns = ["印花税","注册资本","增值税",]
#     for c in columns:
#         df_train,df_test = knn_miss(df_train,df_test,c)
    if istrain:
        df = df_train.copy()
    else:
        df = df_test.copy()
    
    df["经营范围数量"] = df["经营范围"].apply(business)
    df["企业缴税"] = df["增值税"]+df["企业所得税"]+df["印花税"]+df["教育费"]+df["城建税"]
    df["增值税/企业缴税"] = df["增值税"]/df["企业缴税"]
    df["企业所得税/企业缴税"] = df["企业所得税"]/df["企业缴税"]
    df["印花税/企业缴税"] = df["印花税"]/df["企业缴税"]
    df["教育费/企业缴税"] = df["教育费"]/df["企业缴税"]
    df["城建税/企业缴税"] = df["城建税"]/df["企业缴税"]
    df["教育费/城建税"] = df["教育费"]/df["城建税"]
    df["教育费/增值税"] = df["教育费"]/df["增值税"]
    df["企业所得税/投资总额"] = df["企业所得税"]/df["投资总额"]
    df["增值税/经营范围数量"] = df["增值税"]/df["经营范围数量"]
    df["注册资本/增值税"] = df["注册资本"]/df["增值税"]
        
    if istrain:
        train_lable = pd.read_csv(data + "train_label.csv", encoding='utf-8')
        df = pd.merge(train_lable,df,on=['ID'],how='left')    
        df.to_csv(data+"train_deal.csv",index=None,encoding='utf_8_sig')
    else:
        df.to_csv(data+"test_deal.csv",index=None,encoding='utf_8_sig')

def xgb_model():
    global params_xgb
    X_data,test_x = data_review()
    y_label = X_data["Label"]
    
    sub = test_x[["ID"]].copy()
    
    drop_labels= ["ID","经营期限至","邮政编码","核准日期","行业代码","注销时间","经营期限自","成立日期","经营范围","Label"] #+ [ "登记机关", "企业状态", "企业类别", "管辖机关"]
    X_data.drop(drop_labels,axis=1,inplace=True)
    test_x.drop(drop_labels,axis=1,inplace=True)

    res = []
    kf = KFold(n_splits = 5, shuffle=True, random_state=1000)
    score = []
    for i, (train_index, test_index) in enumerate(kf.split(X_data)):
        print('第{}次训练...'.format(i+1))
       
        train_data = X_data.iloc[train_index]
        train_label = y_label.iloc[train_index]
          
        valid_data = X_data.iloc[test_index]
        valid_label = y_label.iloc[test_index]

        xgb_train = xgb.DMatrix(train_data, label=train_label)
        xgb_valid = xgb.DMatrix(valid_data, valid_label)
        evallist = [(xgb_train, 'train'),(xgb_valid, 'eval')]
        cgb_model = xgb.train(params_xgb, xgb_train, num_boost_round=3000, evals=evallist, verbose_eval=300, early_stopping_rounds=400)
        
        vaild_pre = cgb_model.predict(xgb_valid, ntree_limit=cgb_model.best_ntree_limit)
        auc = myFevalAuc(vaild_pre,valid_label)
        if auc<0.928:
            continue
        score.append(auc)
        print("score: ",auc)
        xgb_test = xgb.DMatrix(test_x)
        preds = cgb_model.predict(xgb_test, ntree_limit=cgb_model.best_ntree_limit)
        res.append(MinMaxScaler().fit_transform(preds.reshape(-1, 1)))
        print("\n")
#         plot_importance(cgb_model,max_num_features=40)
#         plt.show()
    print("平均得分: ",sum(score)/len(score))
    res = np.array(res)
    res = res.mean(axis=0)
    sub["Label"] = res
    sub.to_csv('submission.csv', index=False,encoding='utf-8')      

def myFevalAuc(preds, xgb_label):
    fpr, tpr, _ = roc_curve(xgb_label, preds, pos_label=1)
    return auc(fpr, tpr)

def read_x():
#     workbook=xlrd.open_workbook("data/auc.xls")
    df = pd.read_csv('data/auc1.csv', encoding='utf-8',names=['A', 'B'])
    df = df.sort_values(by="B",ascending=False)
    print(df.head(2000))
    
if __name__ == "__main__":
    print("start")
    params_xgb = {
        'booster':'gbtree',
        'objective': 'binary:logistic',
        'eval_metric':'auc',
        
#         'gamma':0.2,        #损失下降多少才分裂
        'min_child_weight':3,
        'max_depth':8,
        'lambda':3,     #控制模型复杂度的权重值的L2曾泽化参数，参数越大越不容易过拟合
        'subsample':0.85,    #随机采样的训练样本
        'colsample_bytree':0.7, #生成树时特征采样比例
        'colsample_bylevel':0.75, #表示树的每层划分的特征采样比例
        'eta': 0.03,
        
        'tree_method':'hist', #表示树的构建方法，确切来说是切分点的选择算法，'gpu_hist' or 'gpu_exact'包括贪心算法，近似贪心算法，直方图算法
        'seed':1000,
        'silent': 0,    #信息输出设置成1则没有信息输出
        'nthread':12
        }
#     data_deal()
    xgb_model()
#     data_review()
#     read_x()
    
    
    
    
    