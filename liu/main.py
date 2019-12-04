# coding=utf-8
#!/usr/bin/python3
'''
单模0.93239
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
import numpy as np
import xgboost as xgb
from xgboost import plot_importance
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import auc, roc_curve, roc_auc_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt 
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

data = "data/"
def get_leaf(x_train, y_train, x_val,mln):
    x_train, y_train, x_val = map(np.array, [x_train, y_train, x_val])
    x_train = x_train.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)
    x_val = x_val.reshape(-1, 1)
    m = DecisionTreeClassifier(min_samples_leaf=9, max_leaf_nodes=mln)
    m.fit(x_train, y_train)
    return m.apply(x_val), roc_auc_score(y_train, m.predict_proba(x_train)[:, 1])

def change_time(x):
    try:
        months, days = str(x)[:5].split(":")
        result = int(months) * 60 + int(days)
    except ValueError:
        result = -1
    return result

def data_review():
    
#     if not os.path.exists(data+"train_use.csv"):
    if True:
        data_deal(istrain=True)
        df_train = pd.read_csv(data + "train_deal.csv", encoding='utf-8',low_memory=False)
        data_deal(istrain=False)
        df_test = pd.read_csv(data + "test_deal.csv", encoding='utf-8',low_memory=False)

        target_name= "Label"
        id_name = "ID"
        df_dt = pd.concat([df_train, df_test], sort=False, ignore_index=True)

               
        cat_features = {'企业类型': 49, 
                        '登记机关': 14, '邮政编码': 80, '行业代码': 248, '行业门类': 18, '管辖机关': 14,  '经营范围': 60, 
                        '企业状态': 5, '企业类别': 5, 
                        '经营期限自':25, '经营期限至':25, '成立日期':25, '核准日期':25,
                        '注销时间': 21,
                        }
        time_features = ['经营期限自', '经营期限至', '成立日期', '核准日期',"注销时间"]
        for tf in time_features:
            df_dt[tf] = df_dt[tf].apply(lambda x: change_time(x))
            
        for feature,leaf_num in cat_features.items():
            if leaf_num > 25:
                leaf_num = 25
            if feature not in time_features:
                df_dt[feature] = LabelEncoder().fit_transform(df_dt[feature].astype(np.str)) 
            df_dt[feature], sc = get_leaf(df_dt[feature].iloc[:-len(df_test)], df_dt[target_name].iloc[:-len(df_test)], df_dt[feature],leaf_num)

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
            count_table.drop([ 0, 1, "rate_pos", "rate_neg"], axis=1, inplace=True)
            count_table.columns = [feature + i for i in count_table.columns]
            df_dt = df_dt.merge(count_table.reset_index(), left_on=feature, right_on=feature, how="left").drop(feature, axis=1)

        # 处理综合数值型数据
        before_feature = list(set([var for var in df_dt.columns if "年初数" in var]))
        before_feature.sort()
        after_feature = list(set([var for var in df_dt.columns if "年末数" in var]))
        after_feature.sort()
        delta_feature = df_dt[after_feature].values - df_dt[before_feature].values
        delta_feature = pd.DataFrame(delta_feature, columns=[var.strip("年初数") + "delta" for var in before_feature])
        df_dt = df_dt.join(delta_feature)
     
        train_d = df_dt.iloc[:-len(df_test)]
        test_d = df_dt.iloc[-len(df_test):]
        
        train_d.to_csv(data+"train_use.csv",index=None,encoding='utf-8')
        test_d.to_csv(data+"test_use.csv",index=None,encoding='utf-8')
    
    train_d = pd.read_csv(data+"train_use.csv",encoding='utf-8',low_memory=False)
    test_d = pd.read_csv(data+"test_use.csv",encoding='utf-8',low_memory=False)
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
    df_train = pd.read_csv(data + "train.csv", encoding='utf-8',low_memory=False)

#     #去除重复数据
#     features = [var for var in df_train.columns if var not in ["ID","Label"]]
#     df = df_train.drop_duplicates(subset=features, keep=False)["ID"].values
#     baddf = df_train[~df_train["ID"].isin(df)]
#     train_lable = pd.read_csv(data + "train_label.csv", encoding='utf-8')
#     baddf = pd.merge(train_lable,baddf,on=['ID'],how='right')
#     bad_ids = baddf[baddf["Label"]==0]["ID"].values
#     df_train = df_train[~df_train["ID"].isin(bad_ids)]

    if istrain:
        df = df_train.copy()
    else:
        df = pd.read_csv(data + "test.csv", encoding='utf-8',low_memory=False)
    

    df["经营范围"] = df["经营范围"].apply(lambda x: x.count(","))
    df["经营具体数目"] =  df["经营范围"]
    df["企业缴税"] = df["增值税"]+df["企业所得税"]+df["印花税"]+df["教育费"]+df["城建税"]
    df["增值税/企业缴税"] = df["增值税"]/df["企业缴税"]
    df["企业所得税/企业缴税"] = df["企业所得税"]/df["企业缴税"]
    df["印花税/企业缴税"] = df["印花税"]/df["企业缴税"]
    df["教育费/企业缴税"] = df["教育费"]/df["企业缴税"]
    df["城建税/企业缴税"] = df["城建税"]/df["企业缴税"]
    df["教育费/城建税"] = df["教育费"]/df["城建税"]
    df["教育费/增值税"] = df["教育费"]/df["增值税"]
    
    df["企业所得税/投资总额"] = df["企业所得税"]/df["投资总额"]
    df["企业所得税/注册资本"] = df["企业所得税"]/df["注册资本"]
    df["增值税/注册资本"] = df["增值税"]/df["注册资本"]
    df["增值税/投资总额"] = df["增值税"]/df["投资总额"]
    df["企业缴税/投资总额"] = df["企业缴税"]/df["投资总额"]
    df["企业缴税/注册资本"] = df["企业缴税"]/df["注册资本"]
    
    df["企业缴税/经营具体数目"] = df["企业缴税"]/df["经营具体数目"]
    df["投资总额/经营具体数目"] = df['投资总额']/df['经营具体数目']
    df["注册资本/经营具体数目"] = df['注册资本']/df['经营具体数目']
    
    df["投资总额/注册资本"] = df['投资总额']/df['注册资本']
        
    if istrain:
        train_lable = pd.read_csv(data + "train_label.csv", encoding='utf-8')
        df = pd.merge(train_lable,df,on=['ID'],how='right')    
        df.to_csv(data+"train_deal.csv",index=None,encoding='utf_8_sig')
    else:
        df.to_csv(data+"test_deal.csv",index=None,encoding='utf_8_sig')

def xgb_model():
    params_xgb = {
        'booster':'gbtree',
        'objective': 'binary:logistic',
        'eval_metric':'auc',
        'min_child_weight':3,
        "min_child_samples":50,
        "max_bin":60,
        'colsample_bytree':0.7, #生成树时特征采样比例
        'colsample_bylevel':0.7, #表示树的每层划分的特征采样比例
        'eta': 0.01,
        'max_depth':5,
        'lambda':2,     #控制模型复杂度的权重值的L2曾泽化参数，参数越大越不容易过拟合
        'tree_method':'hist', #表示树的构建方法，确切来说是切分点的选择算法，'gpu_hist' or 'gpu_exact'包括贪心算法，近似贪心算法，直方图算法
        'seed':1000,
        'silent': True,    #信息输出设置成1则没有信息输出
        'nthread':12
        }
    X_data,test_x = data_review()
    y_label = X_data["Label"]
    print(X_data.shape)

    sub = test_x[["ID"]].copy()
    
    drop_labels= ["ID","Label"] 
    X_data.drop(drop_labels,axis=1,inplace=True)
    test_x.drop(drop_labels,axis=1,inplace=True)

    res = []
    score = []
#     from lightgbm import LGBMClassifier
#     params_initial = {
#     'objective': 'binary',
#     "learning_rate":0.01,
#     'metric': 'auc',
#     'num_leaves': 32,
#     'max_bin': 50,
#     'max_depth': 5,
#     'min_child_samples': 50,
#     'min_child_weight': 3,
#     'n_jobs': -1,
#     }
    for iter in range(100):
        print("iter %02d" % (iter+1))
        train_data, valid_data, train_label, valid_label = train_test_split(X_data, y_label, test_size=0.3, random_state=100+iter)
#         x_train, x_test, y_train, y_test = train_test_split(X_data, y_label, test_size=0.4, random_state=iter)
#         clf = LGBMClassifier(n_estimators=1000, **params_initial)
#         clf.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=100, early_stopping_rounds=500)
        
        xgb_train = xgb.DMatrix(train_data, label=train_label,silent=True)
        xgb_valid = xgb.DMatrix(valid_data, valid_label,silent=True)
        evallist = [(xgb_train, 'train'),(xgb_valid, 'eval')]
        cgb_model = xgb.train(params_xgb, xgb_train, num_boost_round=3000, evals=evallist, verbose_eval=300, early_stopping_rounds=500)
        vaild_pre = cgb_model.predict(xgb_valid, ntree_limit=cgb_model.best_ntree_limit)
        sauc = myFevalAuc(vaild_pre,valid_label)
        score.append(sauc)
        print("iter %02d" % (iter+1),"score: ",sauc)
        
#         importance = cgb_model.get_score(importance_type="weight")
#         tuples = [(k, importance[k]) for k in importance]
#         tuples = sorted(tuples, key=lambda x: x[1])
#         for i in tuples:
#             print(i)
#         print(len(tuples))
#         plot_importance(cgb_model,max_num_features=50)
#         plt.show()

        xgb_test = xgb.DMatrix(test_x,silent=True)
        preds = cgb_model.predict(xgb_test, ntree_limit=cgb_model.best_ntree_limit)
        res.append(MinMaxScaler().fit_transform(preds.reshape(-1, 1)))
        print("\n")

    print("############################")
    print("平均得分: ",sum(score)/len(score)," 模型数量: ",len(score))
    res = np.array(res)
    res = res.mean(axis=0)
    sub["Label"] = res
    sub.to_csv('submission.csv', index=False,encoding='utf-8')      

def myFevalAuc(preds, xgb_label):
    fpr, tpr, _ = roc_curve(xgb_label, preds, pos_label=1)
    return auc(fpr, tpr)
    
if __name__ == "__main__":
    print("start")
    df1 = pd.read_csv("user_data/lgb.csv", encoding='utf-8')
    df1.rename(columns={'Label':'lgb'},inplace=True)
    df2 = pd.read_csv("user_data/122.csv", encoding='utf-8')
    df2.rename(columns={'Label':'122'},inplace=True)
    df3 = pd.read_csv("user_data/sub.csv", encoding='utf-8')
    df3.rename(columns={'Label':'sub'},inplace=True)
    df1 = pd.merge(df1,df2,on=['ID'],how='right')
    df1 = pd.merge(df1,df3,on=['ID'],how='right')
    print(df1.corr(method='pearson', min_periods=1))
#     df1["Label"] = 0.55*df1["Label1"]+0.45*df1["Label2"]
#     df1.drop(["Label1","Label2"],axis=1,inplace=True)
#     df1.to_csv('com_xgb_lgb2.csv', index=False,encoding='utf-8') 
    print(df1.head())

#     data_deal()
#     xgb_model()
#     data_review()
#     read_x()
    
    
        
    
    