import numpy as np
import json
import re

with open('model.txt', 'r') as f:
    coff = f.read().split('\n')[17:]

res = []
alpha = 0.1
for i in coff[:-1]:
    i = str(i).split(' ')
    if eval(i[-1]) > alpha:
       res.append(i[0][:-1])
print(res)

with open('xgb.txt', 'r') as f:
    xgb = f.read()
di = {}
for id,ix in enumerate(xgb.split('booster')[1:]):
    di[id] = re.findall(r'f\d{1,2}',ix)
print(di)