import numpy as np
import pandas as pd 
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

#getting raw data
x=np.loadtxt("data/train_x.csv", delimiter=",")
y=np.loadtxt("data/train_y.csv", delimiter=",")

clf=linear_model.SGDRegressor(shuffle=True, n_iter=5, eta0=0.00000000001, power_t=0.10)
clf.fit(x,y)

predictions = clf.predict(x)
pred = predictions.astype(int)
np.savetxt("results.csv", pred, delimiter=",")



