import pandas as pd
import numpy as np
import csv
from sklearn import linear_model


with open('train.csv') as csvfile:
	data = csv.reader(csvfile,delimiter=',')    	#reading data from csv file 
	train_data = []						#for features
	train_target = []						#for target variables
	dic = {'ac bus':100.0,'auto rickshaw':40.0,'mini bus':30.0,'taxi non ac':50.0,'taxi ac':80.0,'bus':60.0,'metro':90.0,'vehicle_type':0}
	
	
	for row in data:
		 l1 = row[2:6]
		 l1 = [float(i) for i in l1]
		 l3  = row[7:11]
		 l3 = [float(j) for j in l3]
		 train_data.append(l3)
		 #[dic[row[6].lower()]]
		# train_data.append(l1)
		# train_data.append(row[7:11])
		#lc = row[6].lower()
		#train_data.append(dic[lc])
		#train_data.append(row[7:11])
		 train_target.append(float(row[11]))
	#test_idx = [0]
	#train_data = np.delete(train_data,test_idx,axis=0)
	#train_target = np.delete(train_target,test_idx)
	#print(train_data[0])
	print(train_target[0:20])
	print ("The train data has",train_data.shape)
	print ("The  target data has",train_target.shape)
	
	
	test_idx  = np.arange(1001)        
	Train_target = np.delete(train_target,test_idx) #target has next 10,000
	Train_data   = np.delete(train_data,test_idx,axis=0) #data has next 10,000
	
	Test_data=train_data[0:1000] #test_target 
	Test_target=train_target[0:1000]
	
	
	
	
	
	reg = linear_model.LinearRegression()
	
	#from sklearn.linear_model import Lasso
	#reg = Lasso(alpha=0.0001,precompute=True,max_iter=1000,positive=True, random_state=9999, selection='random')
	reg.fit(Train_data,Train_target)
	z = reg.score(Train_data,Train_target)
	print(z)
	"""
	from sklearn.svm import SVR
	reg = SVR(kernel='linear',C=1)
	reg.fit(Train_data,Train_target)
	"""
	
	
	
	prediction = reg.predict(Test_data)
	print(reg.coef_)
	print(prediction[0:20])
	#print(np.mean((prediction-Test_target)**2))
	from sklearn.metrics import r2_score
	print(r2_score(Test_target,prediction))
	
		
