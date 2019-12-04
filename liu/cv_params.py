# coding=utf-8
#!/usr/bin/python3
'''
Created on 2019年11月26日

@author: liushouhua
'''
import pandas as pd
from sklearn.model_selection import KFold
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve
from liu.main2 import data_review
def myFevalAuc(preds, xgb_label):
    fpr, tpr, _ = roc_curve(xgb_label, preds, pos_label=1)
    return auc(fpr, tpr)

def grid_search(param_grid):
    params_xgb = {
        'booster':'gbtree',
        'objective': 'binary:logistic',
        'eval_metric':'auc',
        'min_child_weight':3,
        "min_child_samples":50,
        "max_bin":70,
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
    
#     data = pd.read_csv( "data.csv", encoding='utf-8',low_memory=False)
#     X_data = data.drop(["Label", "ID"], axis=1).values[:-9578]
#     y_label = data.Label.values[:-9578]
    
    
    X_data, _ = data_review()
    y_label = X_data["Label"]
    drop_labels= ["ID","Label"] 
    X_data.drop(drop_labels,axis=1,inplace=True)
    
    best_score = -1
    best_param = {}
    iters = 0
    kf = KFold(n_splits = 5, shuffle=True, random_state=520)
    for i in param_grid[0][1]:
        p_name0 = param_grid[0][0]
        for j in param_grid[1][1]:
            p_name1 = param_grid[1][0]
            for k in param_grid[2][1]:
                iters += 1
                p_name2 = param_grid[2][0]
                params_xgb[p_name0]=i
                params_xgb[p_name1]=j
                params_xgb[p_name2]=k
                score = []
                for _, (train_index, test_index) in enumerate(kf.split(X_data)):
                   
                    train_data = X_data.iloc[train_index]
                    train_label = y_label.iloc[train_index]
                    valid_data = X_data.iloc[test_index]
                    valid_label = y_label.iloc[test_index]
                    xgb_train = xgb.DMatrix(train_data, label=train_label)
                    xgb_valid = xgb.DMatrix(valid_data, valid_label)
                    evallist = [(xgb_train, 'train'),(xgb_valid, 'eval')]
                    cgb_model = xgb.train(params_xgb, xgb_train, num_boost_round=3000, evals=evallist, verbose_eval=0, early_stopping_rounds=500)
                    vaild_pre = cgb_model.predict(xgb_valid, ntree_limit=cgb_model.best_ntree_limit)
                    sauc = myFevalAuc(vaild_pre,valid_label)
                    score.append(sauc)
                auc_ave = sum(score)/len(score)
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print("当前得分:",auc_ave," iters: ",iters)
                print(auc_ave,p_name0,params_xgb[p_name0],p_name1,params_xgb[p_name1],p_name2,params_xgb[p_name2])
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                
                if auc_ave >= best_score:
                    best_score = auc_ave
                    best_param = params_xgb
                    print("************************************************************************************")
                    print("score: ",best_score," param: ",best_param)
                    print("************************************************************************************")
                    
    print("##################################")
    print("score: ",best_score," param: ",best_param)
    print("##################################")


if __name__ == "__main__":

    param_grid = [("max_bin",[60, 70, 80]), ("eta",[0.008, 0.01, 0.013 ]), ("lambda",[0.8, 1, 1.5, 2])]
    print()
    grid_search(param_grid)
    
    
    
    