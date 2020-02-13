import os
import re


# os.chdir('./NewsDownload/')
with open('./xgb.txt') as f:
    xgb = f.read()

xgb = xgb.split('booster')[1:2]
# print(xgb)
sel = re.findall('f\d{1,2}',str(xgb))
res= []
for s in sel:
    if s not in res:
        res.append(s)
print('一共{}个'.format(len(res)),',分别为',res)
