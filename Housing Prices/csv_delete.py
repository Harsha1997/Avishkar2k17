import csv
import numpy as np
with open('train.csv', 'r') as f_input, open('actual_train1.csv', 'w') as f_output,open('test.csv','w') as ftest:
	csv_output = csv.writer(f_output, delimiter = ',')
	csv_test = csv.writer(ftest,delimiter=',')
	ps = ['House ID','Builder Name','Date Built','Date Priced','garden','Dock','Capital','Royal Market','Guarding Tower','River','renovation','dining rooms','bedrooms','bathrooms','visit','Sorcerer','blessings','land','Location','Holy tree','Knight\'s house']
	csv_test.writerow(ps)
	for row in csv.reader(f_input, delimiter = ','):
		try:
			if len(row[21]):
				#print(row[21])
				csv_output.writerow(row)   		
		except:
			csv_test.writerow(row)
			continue		
        
