import datetime as dt
import pandas as pd
import csv

df=pd.DataFrame({'date':['2002-04-24 01:30:00.000']})
df['date'] = pd.to_datetime(df['date'])
print((df['date'] - dt.datetime(1970,1,1)).dt.total_seconds())

