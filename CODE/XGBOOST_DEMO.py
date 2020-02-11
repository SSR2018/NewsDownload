import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import graphviz
#找到所有csv文件
listdir = []
for dir in os.listdir('../../'):
    if dir[-3:] == 'csv':
        listdir.append('./'+dir)
#读取文件
df = pd.read_csv(listdir[0],)
print(df.columns)

#建立想，有，
theta = 1
x = df.iloc[1:,2:].values
y = df['pct_chg'].map(lambda x:0 if x < theta else 1).values[:-1]
np.savetxt('./label.txt',y)
print('x:',x.shape,'y:',y.shape)

print('开始预测...')
dtrain = xgb.DMatrix(x,label=y)
dtest = xgb.DMatrix(x)
params = {'booster': 'gbtree',
          'objective': 'binary:logistic',
          'eval_metric': 'auc',
          'max_depth': 4,
          'lambda': 10,
          'subsample': 0.75,
          'colsample_bytree': 0.75,
          'min_child_weight': 2,
          'eta': 0.025,
          'learning_rate':0.05,
          'seed': 0,
          'nthread': 8,
          'silent': 0}

model = xgb.train(params=params,dtrain = dtrain,evals = [(dtrain,'train')]\
                  ,num_boost_round=100,verbose_eval=10)

pre_y = model.predict(dtest,pred_leaf = True)
np.savetxt('./pre.txt',pre_y)
model.dump_model('./xgb.txt')
