import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
from sklearn.tree import  DecisionTreeClassifier
import matplotlib.pyplot as plt

car_seats = pd.read_csv("C:\\Users\\kp\\Pictures\\Assignments\\Decision Trees\\Company_Data.csv",header=0)

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


train_data,test_data = train_test_split(modified_car_seats,test_size = 0.2,random_state=8)

#Decision Tree Model Training and prediction
model_DT = DecisionTreeClassifier(criterion = 'gini',max_depth = 4,min_samples_split=15)
model_DT.fit(train_data[features],train_data[output])
preds = model_DT.predict(test_data[features])


#Plot to analyse variation between the Predicted values and the Actual Values on Test Dataset
pd.crosstab(test_data[output],preds).plot(kind='bar',color=['red','blue'], grid=True)
plt.legend(['Predicted as >= 7.5','Predicted as < 7.5'])
plt.show()

#Plot to analyse variation between the Predicted values and the Actual Values on Train Dataset
pd.crosstab(train_data[output],model_DT.predict(train_data[features])).plot(kind='bar',color=['red','blue'], grid=True)
plt.legend(['Predicted as >= 7.5','Predicted as < 7.5'])
plt.show()
# pd.crosstab(train_data[output],model_DT.predict(train_data[features]))

# Train Accuracy
print("Train Accuracy achieved by DT Algorithm(%): ",np.mean(train_data.Sales == model_DT.predict(train_data[features]))*100)

# Test Accuracy
print("Test Accuracy achieved by DT Algorithm(%): ",np.mean(preds==test_data.Sales)*100)
