import pandas as pd
import numpy as np
import pylab
import seaborn as sns
import scipy.stats as st
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split

corolla=pd.read_csv("C:\\Users\\kp\\Pictures\\Assignments\\Multi Linear Regression\\ToyotaCorolla.csv",header=0,encoding='latin1')

corolla=pd.get_dummies(corolla,columns=['Color','Fuel_Type'])
corolla.drop(['Id','Model','Mfg_Month','Mfg_Year','Guarantee_Period'],axis=1,inplace=True)

# #Dropping records which are too sensitive for the regressor to learn the pattern(Outliers) by Influencer Plot
# corolla.drop(corolla.index[[601]],axis=0,inplace=True)

#Transforming the independant variables values to the Same scale(Feature Scaling)
corolla_scaled=corolla.iloc[:,1:].transform(func='sqrt')
corolla_scaled['Price']=corolla['Price']


features=['Age_08_04', 'KM', 'HP', 'Cylinders', 'Gears', 'Quarterly_Tax', 'Weight', 'Mfr_Guarantee',
          'BOVAG_Guarantee', 'Airco', 'Automatic_airco', 'CD_Player', 'Powered_Windows', 'Backseat_Divider',
          'Color_Black', 'Color_Blue', 'Color_Green', 'Color_Grey', 'Color_Red', 'Color_Silver', 'Fuel_Type_Petrol']
target=['Price']

### Splitting the data into train and test data
train_data,test_data = train_test_split(corolla_scaled,test_size = 0.2,random_state=42) # 20% size


#Model Building

regressor=smf.ols("Price~Age_08_04+ KM+ HP+ Cylinders+ Gears+ Quarterly_Tax+ Weight+ Mfr_Guarantee+ BOVAG_Guarantee+ Airco+ Automatic_airco+ CD_Player+ Powered_Windows+ Backseat_Divider+ Color_Black+ Color_Blue+ Color_Green+ Color_Grey+ Color_Red+ Color_Silver+ Fuel_Type_Petrol",
                  data=train_data).fit()

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
plt.show()

# Residual values
train_residual  = train_pred - train_data['Price']
test_residual  = test_pred - test_data['Price']

# RMSE value for train and test Data
train_rmse = np.sqrt(np.mean(train_residual*train_residual))
test_rmse = np.sqrt(np.mean(test_residual*test_residual))

print("RMSE Value for the Train Dataset: ",train_rmse)
print("RMSE Value for the Test Dataset: ",test_rmse)
print("R-Square Value of the Model(Measure of Fit): ",regressor.rsquared)