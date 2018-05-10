import numpy as np
import pandas as pd 

#getting raw data
pred_y=np.loadtxt("data/results.csv", delimiter=",")
y=np.loadtxt("data/train_y.csv", delimiter=",")

counter=0
for i in range(12459):
	if y[i] == pred_y[i]: 
		counter += 1
print(counter)

		 


