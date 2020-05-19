import pandas as pd
import numpy as np
import pylab
import seaborn as sns
import scipy.stats as st
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.preprocessing import scale

startup=pd.read_csv("C:\\Users\\kp\\Pictures\\Assignments\\Multi Linear Regression\\50_Startups.csv",header=0)

# Scatter plot between the variables along with histograms
sns.pairplot(startup)
plt.show()

startup=pd.get_dummies(startup,columns=['State'])

#Dropping records which are too sensitive for the regressor to learn the pattern(Outliers) by Influencer Plot
startup.drop(startup.index[[49,19,45]],axis=0,inplace=True)

features=['R&D Spend', 'Marketing Spend']
target=['Profit']

modified_startup=startup
modified_startup['Marketing Spend']=startup['Marketing Spend'].transform(func='sqrt')
modified_startup['R&D Spend']=startup['R&D Spend'].transform(func=scale)
# modified_startup['Administration']=startup['Administration'].transform(func=scale)


#Model Building

regressor=smf.ols("Profit~Q('R&D Spend')+Q('Marketing Spend')",data=modified_startup).fit()

#Summary
print(regressor.summary())

preds=regressor.predict(modified_startup[features])
sm.graphics.influence_plot(regressor,figsize=(15,10))
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
plt.scatter(preds,regressor.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("Fitted_Values");plt.ylabel("Residuals")
plt.show()

# Residual values
residuals = preds - modified_startup['Profit']

# RMSE value for the Dataset
rmse_value = np.sqrt(np.mean(residuals*residuals))

print("RMSE Value for the Dataset: ",rmse_value)
print("R-Square Value of the Model(Measure of Fit): ",regressor.rsquared)