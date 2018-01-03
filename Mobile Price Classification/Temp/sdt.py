import csv
import numpy as np
from sklearn import tree
with open('Train.csv') as csvfile:
	data = csv.reader(csvfile,delimiter=',')    #reading data from csv file 
	train_data = []						#for features
	train_target = []						#for target variables
	for row in data:
		train_data.append(row[:20])
		train_target.append(row[20])
	#print(train_data[1])           
	#print(train_target[1])
	
	test_idx  = np.arange(101)          #arranging 0 to 100 indices in test_idx
	#deleting target values(idx) from  target values
	Train_target = np.delete(train_target,test_idx)
	Train_data   = np.delete(train_data,test_idx,axis=0)
	
	#testing
	Test_data = train_data[0:100]
	Test_target = train_target[0:100]
	
	#Training  :: (Decision tree classifier)
	#10 err /50 
		#clfa = tree.DecisionTreeClassifier()
		#clfa.fit(Train_data,Train_target)
	#Prediction 
	#print(Test_target)
	#print(clf.predict(Test_data))
	
	
	
	
	#Training :: (Logistic regression)
	#from sklearn.linear_model import LogisticRegressionCV
	#clfb  = LogisticRegression()
	#clfb.fit(Train_data,Train_target)
		
	#print(Test_target)
	#print(clf.predict(Test_data))
	
	
	"""
	#(KnearestNeighbors)
	from sklearn.neighbors import KNeighborsClassifier
	clf  = KNeighborsClassifier(n_neighbors=50)
	clf.fit(Train_data,Train_target)
	print(Test_target)
	print(clf.predict(Test_data))
	"""
	
	#highest accuracy (svm classifier :: robust against outliers)
	from sklearn import svm
	my_classifier=svm.SVC(kernel='linear',C=1)
	my_classifier.fit(Train_data,Train_target)
	prediction=my_classifier.predict(Test_data)
	
	
	"""#MLPClassifier
	#from sklearn.neural_network import MLPClassifier
	#my_classifier2 = MLPClassifier()
	#my_classifier2.fit(Train_data,Train_target)
	
	#SVRClassifier
	#from sklearn.svm import SVR
	#my_classifier1 = SVR(C=1.0, epsilon=0.2)
	#my_classifier1.fit(Train_data,Train_target)
	
	#RandomForestClassifier
	from sklearn.ensemble import RandomForestClassifier
	rf = RandomForestClassifier(n_estimators=300)
	rf.fit(Train_data,Train_target)
	"""
	

	from sklearn.metrics import accuracy_score
	print(accuracy_score(Test_target,prediction)*100)
	print(Test_target)
	print(my_classifier.predict(Test_data))
	
	
		
