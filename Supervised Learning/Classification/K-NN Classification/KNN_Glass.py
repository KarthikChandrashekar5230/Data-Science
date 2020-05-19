import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

glass = pd.read_csv("C:\\Users\\kp\\Pictures\\Assignments\\KNN\\glass.csv",header=0)

# Normalizing the numerical data
glassfeatures_normalized = pd.DataFrame(data = scale(glass.iloc[:,0:9]), columns = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'])
glassfeatures_normalized['Type']=glass['Type']
glassfeatures_normalized.drop(['Ca'],axis=1,inplace=True)

# Training and Test data using
df_train,df_test = train_test_split(glassfeatures_normalized,test_size = 0.15,random_state=8)

# KNN using sklearn
# for 3 nearest neighbours
knc = KNeighborsClassifier(n_neighbors= 3)

# Fitting with training data
knc.fit(df_train.iloc[:,0:8],df_train.iloc[:,8])
# train accuracy
accuracy_train = np.mean(knc.predict(df_train.iloc[:,0:8])==df_train.iloc[:,8])  #81%
# test accuracy
accuracy_test = np.mean(knc.predict(df_test.iloc[:,0:8])==df_test.iloc[:,8])  #63%


accuracy_list = []
# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and
# storing the accuracy values

for i in range(3, 50, 2):
    knc = KNeighborsClassifier(n_neighbors=i)
    knc.fit(df_train.iloc[:, 0:8], df_train.iloc[:, 8])
    accuracy_train = np.mean(knc.predict(df_train.iloc[:, 0:8]) == df_train.iloc[:, 8])
    accuracy_test = np.mean(knc.predict(df_test.iloc[:, 0:8]) == df_test.iloc[:, 8])
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


#Prepare the Model by selecting appropriate K Nearest Neighbors value from the above graph
knc = KNeighborsClassifier(n_neighbors= 21)

# Fitting with training data
knc.fit(df_train.iloc[:,0:8],df_train.iloc[:,8])

# train accuracy
accuracy_train = np.mean(knc.predict(df_train.iloc[:,0:8])==df_train.iloc[:,8])
# test accuracy
accuracy_test = np.mean(knc.predict(df_test.iloc[:,0:8])==df_test.iloc[:,8])
print("Training Dataset Accuracy for KNN model in %:",accuracy_train*100)
print("Testing Dataset Accuracy for KNN model in %:",accuracy_test*100)