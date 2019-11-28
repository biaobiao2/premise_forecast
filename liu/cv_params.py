# coding=utf-8
#!/usr/bin/python3
'''
Created on 2019年11月26日

@author: liushouhua
'''
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from main import data_review,myFevalAuc

def grid_search(param_grid):
    params_xgb = {
        'booster':'gbtree',
        'objective': 'binary:logistic',
        'eval_metric':'auc',
        
#         'gamma':0.2,        #损失下降多少才分裂
        'min_child_weight':3,
        'max_depth':8,
        'lambda':2,     #控制模型复杂度的权重值的L2曾泽化参数，参数越大越不容易过拟合
        'subsample':0.85,    #随机采样的训练样本
        'colsample_bytree':0.85, #生成树时特征采样比例
        'colsample_bylevel':0.85, #表示树的每层划分的特征采样比例
        'eta': 0.02,
        
        'tree_method':'hist', #表示树的构建方法，确切来说是切分点的选择算法，'gpu_hist' or 'gpu_exact'包括贪心算法，近似贪心算法，直方图算法
        'seed':1000,
        'silent': 0,    #信息输出设置成1则没有信息输出
        'nthread':12
        }
    
    X_data = data_review()    
    drop_labels= ["ID","经营期限至","邮政编码","核准日期","行业代码","注销时间","经营期限自","成立日期","经营范围"]
    X_data.drop(drop_labels,axis=1,inplace=True)
    
    train_data, test_data = train_test_split(X_data, test_size=0.2, random_state=0)

    train_label = train_data["Label"]
    train_data = train_data.drop(["Label"],axis=1)

    valid_label = test_data["Label"]
    valid_data = test_data.drop(["Label"],axis=1)

    xgb_train = xgb.DMatrix(train_data, label=train_label)
    xgb_valid = xgb.DMatrix(valid_data, label=valid_label)
    evallist = [(xgb_train, 'train'),(xgb_valid, 'eval')]
    
    best_score = -1
    best_param = {}
    for i in param_grid[0][1]:
        p_name0 = param_grid[0][0]
        for j in param_grid[1][1]:
            p_name1 = param_grid[1][0]
            for k in param_grid[2][1]:
                p_name2 = param_grid[2][0]
                params_xgb[p_name0]=i
                params_xgb[p_name1]=j
                params_xgb[p_name2]=k
                cgb_model = xgb.train(params_xgb, xgb_train, num_boost_round=3000 , evals=evallist, verbose_eval=600, early_stopping_rounds=400)
                vaild_pre = cgb_model.predict(xgb_valid, ntree_limit=cgb_model.best_ntree_limit)
                auc = myFevalAuc(vaild_pre,valid_label)
                print("@@@@@@")
                print(auc,params_xgb[p_name0],params_xgb[p_name1],params_xgb[p_name2])
                if auc >= best_score:
                    best_score = auc
                    best_param = params_xgb
                    print("---------------------------")
                    print("score: ",best_score," param: ",best_param)
                    print("---------------------------")
                    
    print("##################################")
    print("score: ",best_score," param: ",best_param)
    print("##################################")


if __name__ == "__main__":

    param_grid = {"lambda":[4,5],"eta":[0.015,0.02,0.025],"max_depth":[7,8,9]}
    print()
    grid_search(list(param_grid.items()))
    
    
    
    