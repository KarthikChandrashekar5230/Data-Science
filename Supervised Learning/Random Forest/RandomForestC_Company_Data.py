import pandas as pd
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

car_seats = pd.read_csv("C:\\Users\\kp\\Pictures\\Assignments\\Random Forests\\Company_Data.csv",header=0)

sales_category=[]

for value in car_seats['Sales']:
    if (value >= 7.5) :
        sales_category.append('Greater than or Equal to 7.5')
    else:
        sales_category.append('Lesser than 7.5')

modified_car_seats=car_seats.drop(['Sales'],axis=1)
modified_car_seats['Sales']=pd.Series(sales_category)

encoder=preprocessing.LabelEncoder()
modified_car_seats['ShelveLoc']=encoder.fit_transform(modified_car_seats['ShelveLoc'])
modified_car_seats['Education']=encoder.fit_transform(modified_car_seats['Education'])
modified_car_seats['Urban']=encoder.fit_transform(modified_car_seats['Urban'])
modified_car_seats['US']=encoder.fit_transform(modified_car_seats['US'])

features=modified_car_seats.columns[0:10]
output=modified_car_seats.columns[10]


#Random Forest Model Building and Predictions on the Entire Dataset as OOB_Score set to True which
#accounts for predicting missed 1/3th of the data samples in the Bootstrap sample for every Tree which is
#Similar to making each Tree to predict for unseen samples apart from its respective Bootstrap Sample
#OOB_Score tells the performance of the Classifier
model_rf = RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=61,criterion="gini",bootstrap=True,
                                  max_depth=13, min_samples_split=4,random_state=8)

model_rf.fit(modified_car_seats[features],modified_car_seats[output])
modified_car_seats['Predictions'] = model_rf.predict(modified_car_seats[features])
print("RandomForest Classifier performance based on OOB_Score is: ",model_rf.oob_score_)

print(pd.crosstab(modified_car_seats['Sales'],modified_car_seats['Predictions']))


#Plot to analyse variation between the Predicted values and the Actual Values on the Dataset
pd.crosstab(modified_car_seats['Sales'],modified_car_seats['Predictions']).plot(kind='bar',color=['red','blue'], grid=True)
plt.legend(['Predicted as >= 7.5','Predicted as < 7.5'])
plt.show()


# RF Model Accuracy
print("Model Accuracy achieved by RFC Algorithm(%): ",np.mean(modified_car_seats.Sales == modified_car_seats.Predictions)*100)
