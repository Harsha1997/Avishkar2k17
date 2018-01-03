import pandas as pd
import numpy as np
import csv
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import datetime
def dateparse (time_in_secs):    
    return datetime.datetime.fromtimestamp(float(time_in_secs))

with open('actual_train.csv') as csvfile:
	data = pd.read_csv(csvfile,delimiter=',',parse_dates=True,date_parser=dateparse,na_values='.')
	
	data['Builder Name'].fillna('Nan',inplace=True)
	lbl = preprocessing.LabelEncoder()
	lbl.fit(list(data['Builder Name'].values))
	data['Builder Name'] = lbl.transform(list(data['Builder Name'].values))
	
	
	import datetime 
	
	#print(data['Date Built'].head())
	for x in data['Date Built']:
		df = datetime.datetime.strptime(x, "%m/%d/%Y %I:%M %p").strftime("%m/%d/%Y %H:%M")
		#print(df)
		data['Date Built'].replace(x,df)
	data['Date Built'] = pd.to_datetime(data['Date Built'])
	data['Date Built'] = (data['Date Built'] - data['Date Built'].min())  / np.timedelta64(1,'D')
		#x = pd.to_datetime(x)
		#x = (x - datetime.datetime(1970,1,1))/np.timedelta64(1,'D')
		#print(x)
	#print(data.info())
	
	for y in data['Date Priced']:
		df = datetime.datetime.strptime(y, "%m/%d/%Y %I:%M %p").strftime("%m/%d/%Y %H:%M")
		data['Date Priced'].replace(y,df)
	data['Date Priced'] = pd.to_datetime(data['Date Priced'])
	data['Date Priced'] = (data['Date Priced'] - data['Date Priced'].min())  / np.timedelta64(1,'D')
	print(data.info())
	
	data['garden'].fillna(0,inplace=True)
	data['garden'] = pd.DataFrame(data['garden'])
	data['garden'].fillna(0).rolling(window=2,min_periods=1).max()
	#print(data.info())
	
	data['Dock'].fillna(data['Dock'].mean(),inplace=True)
	data['Capital'].fillna(data['Capital'].mean(),inplace=True)
	data['Royal Market'].fillna(data['Royal Market'].mean(),inplace=True)
	data['Guarding Tower'].fillna(data['Guarding Tower'].mean(),inplace=True)
	data['River'].fillna(data['River'].mean(),inplace=True)
	
	data['renovation'].fillna(0,inplace=True)
	data['renovation'] = pd.DataFrame(data['renovation'])
	data['renovation'].fillna(0).rolling(window=2,min_periods=1).max()
	#print(data.info())
	
	data['dining rooms'].fillna(data['dining rooms'].mean(),inplace=True)
	data['bedrooms'].fillna(data['bedrooms'].mean(),inplace=True)
	data['bathrooms'].fillna(data['bathrooms'].mean(),inplace=True)
	
	data['visit'].fillna(0,inplace=True)
	data['visit'] = pd.DataFrame(data['visit'])
	data['visit'].fillna(0).rolling(window=2,min_periods=1).max()
	#print(data.info())

	data['Sorcerer'].fillna(0,inplace=True)
	data['Sorcerer'] = pd.DataFrame(data['Sorcerer'])
	data['Sorcerer'].fillna(0).rolling(window=2,min_periods=1).max()
	
	
	data['blessings'].fillna(data['blessings'].mean(),inplace=True)
	
	data['land'].fillna(0,inplace=True)
	lbl = preprocessing.LabelEncoder()
	lbl.fit(list(data['land'].values))
	data['land'] = lbl.transform(list(data['land'].values))
	
	data['Location'].fillna(0,inplace=True)
	lbl = preprocessing.LabelEncoder()
	lbl.fit(list(data['Location'].values))
	data['Location'] = lbl.transform(list(data['Location'].values))
	
	data['Holy tree'].fillna(0,inplace=True)
	data['Holy tree'] = pd.DataFrame(data['Holy tree'])
	data['Holy tree'].fillna(0).rolling(window=2,min_periods=1).max()
	#print(data.info())
	
	
	data['Knight\'s house'].fillna(data['Knight\'s house'].mean(),inplace=True)
	y = data['Golden Grains']
	del data['Golden Grains']
	del data['House ID'] 
	print(data.isnull().sum())
	X = data
	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)
	
	"""
	from sklearn.ensemble import RandomForestRegressor
	from sklearn import metrics, cross_validation
	ra=0
	hi=0
	no=0
	for ran in range(1,100):
		for n in range(420,450):
			reg1 = RandomForestRegressor(random_state=ran,n_estimators=n)
			reg1.fit(X ,y)
			prediction1 = reg1.predict(X_test)
			ac=r2_score(y_test,prediction1)
			if hi<ac:
				print(ac," ",ran," ",n)
				hi=ac
				ra = ran
				no = n
	print("R r2_score",hi,"ra",ra,"nn",no)
	
	"""
	from sklearn.neighbors import KNeighborsRegressor
	
	reg2 = KNeighborsRegressor(n_neighbors=49)
	reg2.fit(X ,y)	
	prediction2 = reg2.predict(X_test)
	print("KN",r2_score(y_test,prediction2))
	
	from sklearn import linear_model
	reg3 = linear_model.LinearRegression()
	reg3.fit(X ,y)
	prediction3 = reg3.predict(X_test)
	print("LR",r2_score(y_test,prediction3))
	
	from xgboost.sklearn import XGBRegressor
	xclass = XGBRegressor(learning_rate=0.1,max_depth=20)
	xclass.fit(X ,y)
	prediction5 = xclass.predict(X_test)
	print("Lrr",r2_score(y_test,prediction5))
	#print(data.info())
	
	
	
	
	
