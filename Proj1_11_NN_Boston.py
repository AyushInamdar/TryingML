from sklearn import datasets
from keras.models import Sequential 
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


data = pd.read_csv('BostonHousing.csv')
X = data.iloc[:,0:13]
y = data.iloc[:,13]

ss = StandardScaler()
X = ss.fit_transform(X)

Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.2)

model = Sequential()
model.add(Dense(13,input_dim=13,activation="relu"))
model.add(Dense())
model.compile(loss="mean_squared_error",optimizer="adam",metrics=["accuracy"])

history = model.fit(Xtrain,ytrain,epochs=150,batch_size=10)
ypred=model.predict(Xtest)
ypred=ypred[:,0]

error = np.sum(np.abs(ytest-ypred)) / np.sum(np.abs(ytest)) * 100
print("predicted error is ",error , "%")