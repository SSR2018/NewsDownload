import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import graphviz
#找到所有csv文件

os.chdir('./NewsDownload/')
listdir = []
for dir in os.listdir('./'):
    if dir[-3:] == 'csv':
        listdir.append('./'+dir)
print(listdir)
#读取文件
data = pd.DataFrame({})
for li in listdir:
    df = pd.read_csv(li)
    # print(df.columns)
    #建立
    theta = df.pct_chg.abs().mean()
    print(theta)
    df.pct_chg =  df['pct_chg'].map(lambda x:0 if abs(x) < theta else 1)
    data = pd.concat([data,df])
y = data.pct_chg[:-1].values
# x = data.copy(deep=True)
# x = x.drop(['pct_chg'],axis=1)
x = data.iloc[1:,2:].values
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
                  ,num_boost_round=50,verbose_eval=10)

pre_y = model.predict(dtest,pred_leaf = True)
np.savetxt('./pre.txt',pre_y)
model.dump_model('./xgb.txt')
