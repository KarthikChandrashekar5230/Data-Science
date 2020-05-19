import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn import metrics

df_train = pd.read_csv("C:\\Users\\kp\\Pictures\\Assignments\\Naive Bayes\\SalaryData_Train.csv",header=0)
df_test = pd.read_csv("C:\\Users\\kp\\Pictures\\Assignments\\Naive Bayes\\SalaryData_Test.csv",header=0)
string_columns=["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]

categorical_data = preprocessing.LabelEncoder()
for i in string_columns:
    df_train[i] = categorical_data.fit_transform(df_train[i])
    df_test[i] = categorical_data.fit_transform(df_test[i])

column_names = df_train.columns
train_X = df_train[column_names[0:13]]
train_y = df_train[column_names[13]]
test_X = df_test[column_names[0:13]]
test_y = df_test[column_names[13]]

#Gaussian Naive Bayes' Classifier used for Features which are in the form of continuous datatype
gaussian_nbc = GaussianNB()
#Multinomial Naive Bayes' Classifier used for Features which in the form of continuous datatype but mainly used for Document Classification
multinomial_nbc = MultinomialNB()

pred_gaussian = gaussian_nbc.fit(train_X,train_y).predict(test_X)
confusion_matrix(test_y,pred_gaussian)
print ("Gaussian Naive Bayes model accuracy(in %):",metrics.accuracy_score(test_y, pred_gaussian)*100)

pred_multinomial = multinomial_nbc.fit(train_X,train_y).predict(test_X)
confusion_matrix(test_y,pred_multinomial)
print("MultiNomial Naive Bayes model accuracy(in %):",metrics.accuracy_score(test_y, pred_multinomial)*100)