import csv
import numpy as np
from sklearn import tree
with open('Train.csv') as csvfile:
	data = csv.reader(csvfile,delimiter=',')
	train_data = []
	train_target = []
	for row in data:
		train_data.append(row[:20])
		train_target.append(row[20])
	#print(train_data[1])
	#print(train_target[1])
	
	test_idx  = np.arange(101)
	#deleting target values(idx) from  target values
	Train_target = np.delete(train_target,test_idx)
	Train_data   = np.delete(train_data,test_idx,axis=0)
	
	#testing
	Test_data = train_data[0:100]
	Test_target = train_target[0:100]
	"""
	#Training
	#10 err /50
	clf = tree.DecisionTreeClassifier()
	clf.fit(Train_data,Train_target)
	
	#4.Prediction 

	print(Test_target)
	print(clf.predict(Test_data))
	"""
	"""
	from sklearn.linear_model import LogisticRegression
	clf  = LogisticRegression()
	clf.fit(Train_data,Train_target)
	print(Test_target)
	print(clf.predict(Test_data))
	"""
	#Knearest Acc:4 err/50
	from sklearn.neighbors import KNeighborsClassifier
	clf  = KNeighborsClassifier(n_neighbors=15)
	clf.fit(Train_data,Train_target)
	print(Test_target)
	print(clf.predict(Test_data))
	
	
		
