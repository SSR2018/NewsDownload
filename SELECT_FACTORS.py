import numpy as np
import json
import re
import os

os.chdir('./NewsDownload/')
with open('model.txt', 'r') as f:
    coff = f.read().split('\n')[32:]

res = []
alpha = 0.1
for i in coff[:-1]:
    i = str(i).split(' ')
    isTrue = 1
    for j in i[1:]:
        if eval(j) < alpha:
            isTrue = 0
    if isTrue:
       res.append(i[0][:-1])
print(res)

# with open('xgb.txt', 'r') as f:
#     xgb = f.read()
# di = {}
# for id,ix in enumerate(xgb.split('booster')[1:]):
#     di[id] = re.findall(r'f\d{1,2}',ix)
# print(di)