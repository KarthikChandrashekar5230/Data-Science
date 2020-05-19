import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

train_salary = pd.read_csv("C:\\Users\\kp\\Pictures\\Assignments\\Support Vector Machines\\SalaryData_Train(1).csv",
                           header=0)
test_salary = pd.read_csv("C:\\Users\\kp\\Pictures\\Assignments\\Support Vector Machines\\SalaryData_Test(1).csv",
                          header=0)


def data_visualization(train_salary):
    sb.pairplot(data=train_salary)
    plt.show()
    # Pair-Plot shows that the Numerical features present in the dataset are not highly correlated and hence all these features can be used
    # in our Model Building

    pd.crosstab(train_salary.Salary, train_salary.workclass).plot(kind="bar")
    plt.show()
    # The plot clearly says that there is a Significant Variation in the Salary for the WorkClass Private rather than Other
    # Workclassess comparatively

    return None


# data_visualization(train_salary)


# Data Preprocessing for both Train and Test Datasets

def data_preprocessing(dataframe):
    dataframe["Salary>50K"] = 0
    dataframe.loc[dataframe.Salary == ' >50K', "Salary>50K"] = 1
    dataframe = dataframe.reindex(
        columns=['Salary>50K', 'age', 'workclass', 'education', 'educationno', 'maritalstatus', 'occupation',
                 'relationship', 'race', 'sex', 'capitalgain', 'capitalloss', 'hoursperweek', 'native', 'Salary'])

    dummies = pd.get_dummies(
        dataframe[['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 'race', 'sex', 'native']])

    dataframe.drop(['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 'race', 'sex', 'native',
                    'educationno', 'Salary'], axis=1, inplace=True)
    dataframe = pd.concat([dataframe, dummies], axis=1)

    # Standardizing Each Column Values
    standard_dataframe = pd.DataFrame(data=scale(dataframe.iloc[:, 1:]), columns=dataframe.columns[1:])
    standard_dataframe['Salary>50K'] = dataframe['Salary>50K']

    return standard_dataframe


standard_train_salary = data_preprocessing(train_salary)
standard_test_salary = data_preprocessing(test_salary)

features=standard_train_salary.columns[0:101]
target=standard_train_salary.columns[101]


#Model Building

model_svm=SVC(C=5.0,kernel = "rbf",gamma='auto',class_weight='balanced',random_state=8)
model_svm.fit(standard_train_salary[features],standard_train_salary[target])
train_preds = model_svm.predict(standard_train_salary[features])
test_preds= model_svm.predict(standard_test_salary[features])

# confusion matrix using pandas CrossTab
print("Confusion Matrix: \n",confusion_matrix(standard_train_salary[target],train_preds))

#Classification Report on Model's Performance for individual Classes
print("Classification Report of SVM Model: \n",classification_report(standard_train_salary[target],train_preds))


# Train Accuracy
print("Accuracy achieved on Training Dataset by Support Vector Machines (%): ",100*(np.mean(standard_train_salary[target]==train_preds)))

# Test Accuracy
print("Accuracy achieved on Test Dataset by Support Vector Machines (%): ",100*(np.mean(standard_test_salary[target]==test_preds)))
#Tried Different SVM's hyperparameter Settings but it didn't improve Accuracy rate due to imbalanced Dataset,this is the Optimal Accuracy Levels that i can get without Overfitting the data