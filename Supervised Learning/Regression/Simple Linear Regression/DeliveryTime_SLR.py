import pandas as pd
import numpy as numpy
import matplotlib.pyplot as mlpt
from sklearn.linear_model import LinearRegression
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split
from sklearn import metrics


#To read the CSV file
delivery_dataframe=pd.read_csv('C:\\Users\\kp\\Pictures\\Assignments\\Simple Linear Regression\\delivery_time.csv')
#TO derive the number of rows and columns the given data set has
print(delivery_dataframe.shape)

#Assigning the columns and labels to X,Y
X=delivery_dataframe['Delivery Time'].values.reshape(-1, 1)
y=delivery_dataframe['Sorting Time'].values.reshape(-1, 1)

#To split the train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#To train the model
regression=LinearRegression()
regression.fit(X_train,y_train)

#To retrieve the intercept:
print(regression.intercept_)
#For retrieving the slope:
print(regression.coef_)

#For predicting the values
y_pred = regression.predict(X_test)
print(y_pred)

#To create a new data frame.
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df)

#To predict the mean values.
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', numpy.sqrt(metrics.mean_squared_error(y_test, y_pred)))
