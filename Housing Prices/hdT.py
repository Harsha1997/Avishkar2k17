import pandas as pd
import numpy as np
import csv
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import datetime

    
with open('actual_train.csv') as csvfile:
	data = pd.read_csv(csvfile,delimiter=',')
	
	
	
	
	data['Builder Name'].fillna('Nan',inplace=True)
	lbl = preprocessing.LabelEncoder()
	lbl.fit(list(data['Builder Name'].values))
	data['Builder Name'] = lbl.transform(list(data['Builder Name'].values))
	
	
	for x in data['Date Built']:
		df = datetime.datetime.strptime(x, "%m/%d/%Y %I:%M %p").strftime("%m/%d/%Y %H:%M")
		data['Date Built'].replace(x,df)
	data['Date Built'] = pd.to_datetime(data['Date Built'])
	data['Date Built'] = (data['Date Built'] - data['Date Built'].min())  / np.timedelta64(1,'D')
	
	for xyy in data['Date Priced']:
		df = datetime.datetime.strptime(xyy, "%m/%d/%Y %I:%M %p").strftime("%m/%d/%Y %H:%M")
		data['Date Priced'].replace(xyy,df)
	data['Date Priced'] = pd.to_datetime(data['Date Priced'])
	data['Date Priced'] = (data['Date Priced'] - data['Date Priced'].min())  / np.timedelta64(1,'D')
	

	data['garden'].fillna(0,inplace=True)
	data['garden'] = pd.DataFrame(data['garden'])
	data['garden'].fillna(0).rolling(window=2,min_periods=1).max()
	

	data['Dock'].fillna(data['Dock'].mean(),inplace=True)
	data['Capital'].fillna(data['Capital'].mean(),inplace=True)
	data['Royal Market'].fillna(data['Royal Market'].mean(),inplace=True)
	data['Guarding Tower'].fillna(data['Guarding Tower'].mean(),inplace=True)
	data['River'].fillna(data['River'].mean(),inplace=True)
	
	data['renovation'].fillna(0,inplace=True)
	data['renovation'] = pd.DataFrame(data['renovation'])
	data['renovation'].fillna(0).rolling(window=2,min_periods=1).max()
	
	
	data['dining rooms'].fillna(data['dining rooms'].mean(),inplace=True)
	data['bedrooms'].fillna(data['bedrooms'].mean(),inplace=True)
	data['bathrooms'].fillna(data['bathrooms'].mean(),inplace=True)
	
	data['visit'].fillna(0,inplace=True)
	data['visit'] = pd.DataFrame(data['visit'])
	data['visit'].fillna(0).rolling(window=2,min_periods=1).max()
	

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
	
	data['Knight\'s house'].fillna(data['Knight\'s house'].mean(),inplace=True)
	y = data['Golden Grains']
	del data['Golden Grains']
	del data['House ID'] 
	X = data
	
	with open('test.csv') as TestData:
		data1 = pd.read_csv(TestData,delimiter=',')
		
		data1['Builder Name'].fillna('Nan',inplace=True)
		lbl = preprocessing.LabelEncoder()
		lbl.fit(list(data1['Builder Name'].values))
		data1['Builder Name'] = lbl.transform(list(data1['Builder Name'].values))
		
		for x in data1['Date Built']:
			df = datetime.datetime.strptime(x, "%m/%d/%Y %I:%M %p").strftime("%m/%d/%Y %H:%M")
			data1['Date Built'].replace(x,df)
		data1['Date Built'] = pd.to_datetime(data1['Date Built'])
		data1['Date Built'] = (data1['Date Built'] - data1['Date Built'].min())  / np.timedelta64(1,'D')
		
		for xyy in data1['Date Priced']:
			df = datetime.datetime.strptime(xyy, "%m/%d/%Y %I:%M %p").strftime("%m/%d/%Y %H:%M")
			data1['Date Priced'].replace(xyy,df)
		data1['Date Priced'] = pd.to_datetime(data1['Date Priced'])
		data1['Date Priced'] = (data1['Date Priced'] - data1['Date Priced'].min())  / np.timedelta64(1,'D')
		
		
		
		data1['garden'].fillna(0,inplace=True)
		data1['garden'] = pd.DataFrame(data1['garden'])
		data1['garden'].fillna(0).rolling(window=2,min_periods=1).max()
		
		data1['Dock'].fillna(data1['Dock'].mean(),inplace=True)
		data1['Capital'].fillna(data1['Capital'].mean(),inplace=True)
		data1['Royal Market'].fillna(data1['Royal Market'].mean(),inplace=True)
		data1['Guarding Tower'].fillna(data1['Guarding Tower'].mean(),inplace=True)
		data1['River'].fillna(data1['River'].mean(),inplace=True)
		
		data1['renovation'].fillna(0,inplace=True)
		data1['renovation'] = pd.DataFrame(data1['renovation'])
		data1['renovation'].fillna(0).rolling(window=2,min_periods=1).max()
		
		data1['dining rooms'].fillna(data1['dining rooms'].mean(),inplace=True)
		data1['bedrooms'].fillna(data1['bedrooms'].mean(),inplace=True)
		data1['bathrooms'].fillna(data1['bathrooms'].mean(),inplace=True)
		
		data1['visit'].fillna(0,inplace=True)
		data1['visit'] = pd.DataFrame(data1['visit'])
		data1['visit'].fillna(0).rolling(window=2,min_periods=1).max()
		
		data1['Sorcerer'].fillna(0,inplace=True)
		data1['Sorcerer'] = pd.DataFrame(data1['Sorcerer'])
		data1['Sorcerer'].fillna(0).rolling(window=2,min_periods=1).max()
		
		data1['blessings'].fillna(data1['blessings'].mean(),inplace=True)
		
		data1['land'].fillna(0,inplace=True)
		lbl = preprocessing.LabelEncoder()
		lbl.fit(list(data1['land'].values))
		data1['land'] = lbl.transform(list(data1['land'].values))
	
		data1['Location'].fillna(0,inplace=True)
		lbl = preprocessing.LabelEncoder()
		lbl.fit(list(data1['Location'].values))
		data1['Location'] = lbl.transform(list(data1['Location'].values))
	
		data1['Holy tree'].fillna(0,inplace=True)
		data1['Holy tree'] = pd.DataFrame(data1['Holy tree'])
		data1['Holy tree'].fillna(0).rolling(window=2,min_periods=1).max()
		data1['Knight\'s house'].fillna(data1['Knight\'s house'].mean(),inplace=True)
		yx = data1['House ID']
		del data1['House ID'] 
		
	
		from xgboost.sklearn import XGBRegressor
		reg1 = XGBRegressor(learning_rate=0.1,max_depth=20)
		reg1.fit(X,y)
		prediction1 = reg1.predict(data1)
		
		with open('result.csv',"w") as f:
			writer = csv.writer(f)
			ps = ['House ID','Golden Grains']
			writer.writerow(ps)
			
			for x,vc  in  zip(yx,prediction1):
				writer.writerow([x,vc])
					
			
	
	
	
