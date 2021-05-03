from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 
from google.colab import drive     
import numpy as np
import pandas as pd 
drive.mount('/content/drive')
seedValue = 435 #Once model is deployed, random_state arg would be removed from train_test_split. 

#--------------------------------------------------New Cell------------------------------------

def linear_regression(dataSet):
  model = LinearRegression()
  df = dataSet
  trainData, testData = train_test_split(df, test_size=.3, train_size=.7, random_state=seedValue )  #Seed value here
  trainData = trainData.to_numpy()
  testData = testData.to_numpy()
  trainX = trainData[:, 0]
  trainY = trainData[:, 1]
  testX = testData[:, 0]
  testY = testData[:, 1]
  trainX = trainX.reshape((-1, 1))
  testX = testX.reshape((-1, 1))
  model.fit(trainX, trainY)
  pred = model.predict(testX)
  print('model prediction values:', pred)
  r_sq = model.score(testX, testY)
  rmse2 = np.sqrt(mean_squared_error(testY, pred))
  print('R_Squared: ', r_sq)
  print('RMSE: ', rmse2)
  print('intercept: ', model.intercept_)
  print('slope: ', model.coef_)   #Could return all values in an array. Right now just outputs the values. 
         
#--------------------------------------------------New Cell------------------------------------
  
from google.colab import drive  #Optional. Used to create persistance for data. 
drive.mount('/content/drive')   #Optional. Used to create persistance for data.
df = pd.read_csv('')            #Point this towards your dataset. 
linear_regression(df)           #Calls above function. 
