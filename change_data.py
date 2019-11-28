# coding=utf-8
#!/usr/bin/python3
'''
Created on 2019年11月26日

@author: liushouhua
'''
import pandas as pd
import numpy as np
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option("display.max_colwidth",100)
pd.set_option('display.width',1000)
from sklearn.preprocessing import LabelEncoder,  MinMaxScaler
from itertools import combinations
from feature_selsector import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
def get_leaf(x_train, y_train, x_val):
    
    x_train, y_train, x_val = map(np.array, [x_train, y_train, x_val])
    x_train = x_train.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)
    x_val = x_val.reshape(-1, 1)
    m = DecisionTreeClassifier(min_samples_leaf=0.001, max_leaf_nodes=10)
    m.fit(x_train, y_train)
    return m.apply(x_val), roc_auc_score(y_train, m.predict_proba(x_train)[:, 1])

def test():
    train = pd.read_csv("./data/train.csv",low_memory=False).fillna(-1)
    print(train.shape)
    train_label = pd.read_csv("./data/train_label.csv")
    train = train_label.merge(train, on=["ID"],  how="left")
    target_name = "Label"
    id_name = "ID"
#     features = [var for var in train.columns if var not in [target_name, id_name]]
#     train = train.groupby(features, as_index=False)[target_name].mean()
#     train = train[np.abs(train[target_name] - 0.5) == 0.5]

    # 数据重新分箱
    features = ['企业类型', '经营期限至', '登记机关', '企业状态', '邮政编码', '核准日期', '行业代码','注销时间', '经营期限自', '成立日期', '行业门类', '企业类别', '管辖机关']
    
    data = train[features+["ID","Label"]].copy()
    print(data.head())
    for feature in features:
        data[feature] = LabelEncoder().fit_transform(data[feature].astype(np.str))
        data[feature], auc = get_leaf(data[feature], data[target_name], data[feature])
        print(feature, auc)
        if auc > 0.55:
            count_table = pd.pivot_table(data,
                                         index=feature,
                                         columns=target_name,
                                         values=id_name,
                                         aggfunc="count",
                                         fill_value=0)
            
            count_table["total"] = count_table[0] + count_table[1]
            count_table["rate"] = count_table[1] / np.sum(count_table[1]) - count_table[0] / np.sum(count_table[0])
            count_table = count_table.sort_values(by="rate")
            count_table["label"] = range(len(count_table))
            data[feature] = data[feature].replace(count_table.index, count_table.label)
        else:
            data.drop(feature, axis=1)
    return

    print(data.shape)
    for feature in features:
        print(feature)
        count_table = pd.pivot_table(data,
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
        data = data.merge(count_table.reset_index(), left_on=feature, right_on=feature, how="left").drop(feature, axis=1)
        print(data.head())
        return
    
    
#     print(data.shape)
#     features = [var for var in data.columns if var not in [target_name, id_name]]
#     data[features] = MinMaxScaler().fit_transform(data[features])
#     std_train = np.std(data[features].iloc[:-len(test)])
#     std_test = np.std(data[features].iloc[-len(test):])
#     choose = (std_train <= 0.01) & (std_test <= 0.01)
#     try:
#         data = data.drop(choose[choose].index.tolist(), axis=1)
#     except:
#         print("wrong")
#     print(data.shape)
#     features = [var for var in data.columns if var not in [target_name, id_name]]
#     fs = FeatureSelector(data=data[features].iloc[:-len(test)], labels=data[target_name].iloc[:-len(test)])
#     fs.identify_collinear(correlation_threshold= 0.95)
#     data = data.drop(fs.ops['collinear'], axis=1)
#     print(data.shape)
    
if __name__ == "__main__":
    test()
    
    
    
    
    
    
    
    
    
    
    
