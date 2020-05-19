import pandas as pd
import numpy as np
import pylab
import seaborn as sns
import scipy.stats as st
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split

computer=pd.read_csv("C:\\Users\\kp\\Pictures\\Assignments\\Multi Linear Regression\\Computer_Data.csv",header=0)

# Scatter plot between the variables along with histograms
sns.pairplot(computer)
plt.show()

# Scatter plot between the variables along with histograms using Pandas
pd.plotting.scatter_matrix(computer,figsize=(20,15))
plt.show()


computer=pd.get_dummies(computer,columns=['cd','multi','premium'])
computer.drop(['Unnamed: 0'],axis=1,inplace=True)

#Dropping records which are too sensitive for the regressor to learn the pattern(Outliers) by Influencer Plot
computer.drop(computer.index[[5960,4477,3783,900,1101,1610,1524,2042,1835,1688,1700,720,1784,1440,1048,79]],axis=0,inplace=True)

features=['speed', 'hd', 'ram', 'screen', 'ads', 'trend', 'cd_yes', 'multi_yes', 'premium_yes']
target=['price']

### Splitting the data into train and test data
train_data,test_data = train_test_split(computer,test_size = 0.2,random_state=42) # 20% size


#Model Building

regressor=smf.ols("price~speed+hd+ram+screen+ads+trend+cd_yes+multi_yes+premium_yes",data=train_data).fit()

#Summary
print(regressor.summary())

train_pred=regressor.predict(train_data[features])
test_pred=regressor.predict(test_data[features])
sm.graphics.influence_plot(regressor)
plt.show()


#Normality Test for Residuals
#In Order to have a Model's good fit it is important to have the Residuals follow a Normal Distribution Pattern

# Normal Distribution Check using Histogram
plt.figure(figsize=(20,15))
plt.hist(regressor.resid_pearson)
plt.show()

# Normal Distribution Check using Q-Q plot
plt.figure(figsize=(20,15))
st.probplot(regressor.resid_pearson, dist="norm", plot=pylab)
plt.show()

# Homoscedasticity : Error term being same across all values of independent variables
# Fitted Values Vs Residuals
plt.figure(figsize=(15,10))
plt.scatter(train_pred,regressor.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("Fitted_Values");plt.ylabel("Residuals")


# Residual values
train_residual  = train_pred - train_data['price']
test_residual  = test_pred - test_data['price']

# RMSE value for train and test Data
train_rmse = np.sqrt(np.mean(train_residual*train_residual))
test_rmse = np.sqrt(np.mean(test_residual*test_residual))

print("RMSE Value for the Train Dataset: ",train_rmse)
print("RMSE Value for the Test Dataset: ",test_rmse)