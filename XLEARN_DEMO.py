import xlearn
import pandas
import re
import json
from tqdm import tqdm
import numpy as np
import os

os.chdir('./NewsDownload/')
file_name = 'xgb'
with open('./'+file_name+'.txt','r') as f:
    xgb = f.read()
di = {}
for id,ix in enumerate(xgb.split('booster')[1:]):
    leaves = re.findall(r'\d{1,2}:leaf',ix)
    leaves = { eval(ixx[:-5]):idd+1 for idd,ixx in enumerate(leaves)}
    di[id] = leaves
# print(di)
# exit()
label = np.loadtxt('./label.txt')

pre = np.loadtxt('./pre.txt')
print(pre.shape)
libsvm = ''
for id,index in enumerate(pre):
    libsvm_ = ''
    for it,col in enumerate(index):
        libsvm_+='{}:{}:1,'.format(it+1,di[it][col])
    libsvm +=str(label[id])+','+libsvm_+'\n'

with open('./ffm.txt', 'w+') as f:
    f.write(libsvm)

model = xlearn.create_ffm()
param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'metric': 'acc','k':5,'epoch':100}
model.setTrain("./ffm.txt")
model.setTXTModel("./model.txt")
model.fit(param, "./model.out")
