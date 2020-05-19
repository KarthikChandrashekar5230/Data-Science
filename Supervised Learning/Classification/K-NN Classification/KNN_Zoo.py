import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

zoo_animals = pd.read_csv("C:\\Users\\kp\\Pictures\\Assignments\\KNN\\Zoo.csv",header=0)
zoo_animals.drop(['animal name'],axis=1,inplace=True)


# Normalizing the numerical data
zooanimals_normalized = pd.DataFrame(data = scale(zoo_animals.iloc[:,0:16]), columns = ['Hair', 'Feathers', 'Eggs', 'Milk', 'Airborne', 'Aquatic', 'Predator','Toothed', 'Backbone', 'Breathes', 'Venomous', 'Fins', 'Legs', 'Tail','Domestic', 'Catsize'])
zooanimals_normalized['Type']=zoo_animals['type']


# Training and Test data using
df_train,df_test = train_test_split(zooanimals_normalized,test_size = 0.15,random_state=8)

# KNN using sklearn
# for 3 nearest neighbours
knc = KNeighborsClassifier(n_neighbors= 3)

# Fitting with training data
knc.fit(df_train.iloc[:,0:16],df_train.iloc[:,16])
# train accuracy
accuracy_train = np.mean(knc.predict(df_train.iloc[:,0:16])==df_train.iloc[:,16])  #96%
# test accuracy
accuracy_test = np.mean(knc.predict(df_test.iloc[:,0:16])==df_test.iloc[:,16])   #93%

accuracy_list = []
# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and
# storing the accuracy values

for i in range(3, 50, 2):
    knc = KNeighborsClassifier(n_neighbors=i)
    knc.fit(df_train.iloc[:,0:16],df_train.iloc[:,16])
    accuracy_train = np.mean(knc.predict(df_train.iloc[:,0:16])==df_train.iloc[:,16])
    accuracy_test = np.mean(knc.predict(df_test.iloc[:,0:16])==df_test.iloc[:,16])
    accuracy_list.append([accuracy_train, accuracy_test])

plt.figure(figsize=(20,15))
plt.legend(["Training_Set","Testing_Set"])
plt.xlabel("Number of K Nearest Neighbors Selected During Modeling ")
plt.ylabel("Model Performance for Train And Test Datasets")
plt.xticks(np.arange(3,50,2))
plt.title('Plot for Determining appropriate K Nearest Neighbors Value')
# train accuracy plot
plt.plot(np.arange(3,50,2),[i[0] for i in accuracy_list],"bo-")
# test accuracy plot
plt.plot(np.arange(3,50,2),[i[1] for i in accuracy_list],"ro-")
plt.show()



#Prepare the Model by selecting appropriate K Nearest Neighbors value from the above graph Either 5 or 9
knc = KNeighborsClassifier(n_neighbors= 5)

# Fitting with training data
knc.fit(df_train.iloc[:,0:16],df_train.iloc[:,16])

# train accuracy
accuracy_train = np.mean(knc.predict(df_train.iloc[:,0:16])==df_train.iloc[:,16])
# test accuracy
accuracy_test = np.mean(knc.predict(df_test.iloc[:,0:16])==df_test.iloc[:,16])

print("Training Dataset Accuracy for KNN model in %:",accuracy_train*100)
print("Testing Dataset Accuracy for KNN model in %:",accuracy_test*100)