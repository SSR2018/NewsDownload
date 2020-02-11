import tushare as ts
import pandas as pd
import numpy as np

#下载数据
token = 'e8df84bd1b25a8a2a2ceb7edf7ad41f2c3a1d3ec604bb8abd40321f4'
ts.set_token(token)
pro =ts.pro_api()

start_date = '20100101' #开始日期
end_date = '20121231'  #结束日期
ts_code = '601318.SH'  #股票代码

daily_basic = pro.daily_basic(ts_code = ts_code,start_date = start_date,end_date = end_date)
pro_bar  = ts.pro_bar(ts_code = ts_code,adj='qfq',start_date=start_date,end_date =end_date)
df = pd.concat([pro_bar,daily_basic.iloc[:,4:]],axis = 1)

df.to_csv('./'+ts_code[:-3]+'.csv',index = 0)#导入到csv文件 文件名(股票代码.csv)