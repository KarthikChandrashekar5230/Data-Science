import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

bank = pd.read_csv("C:\\Users\\kp\\Pictures\\Assignments\\Logistic Regression\\bank-full.csv", header=0, delimiter=';')


def data_visualization(bank):
    # Bar Plots for Categorical Columns
    sb.countplot(x="y", data=bank,
                 palette="hls")  # Highly Imbalanced Dataset as the values in Target Variable is unevenly distributed
    plt.show()

    pd.crosstab(bank.y, bank.marital).plot(
        kind="bar")  # Proportion of the People Saying No and Saying Yes seems to uniform across
    # each categorical value of the Category
    plt.show()

    pd.crosstab(bank.y, bank.job).plot(
        kind="bar")  # Proportion of the People Saying No and Saying Yes seems to uniform across
    # each categorical value of the Category
    plt.show()

    pd.crosstab(bank.y, bank.education).plot(
        kind="bar")  # Proportion of the People Saying No and Saying Yes seems to uniform across
    # each categorical value of the Category
    plt.show()

    pd.crosstab(bank.y, bank.housing).plot(
        kind="bar")  # Seems like there is a dependability between the Variables Housing and Output
    # y from the Plot
    plt.show()

    return None

# data_visualization(bank)


#Finding NaN Values in the Dataframe and eliminating them

print("Number of NaN/Missing Values present Each Column of the Dataframe: ",bank.isnull().sum())
bank.dropna(inplace=True) # Best Practice though there are no NaN Values present
#There were no NaN values presnt in the Dataframe so the shape of Df remains same even after dropna() has been called

bank["Subscribe"] = 0
bank.loc[bank.y == "yes","Subscribe"] = 1

dummies=pd.get_dummies(bank[['job','marital','education','default','housing','loan','contact','month','poutcome']])
bank.drop(['job','marital','education','default','housing','loan','contact','month','poutcome','day','y'],axis=1,inplace=True)
bank = pd.concat([bank,dummies],axis=1)
bank=bank[['Subscribe','age', 'balance', 'duration', 'campaign', 'pdays', 'previous','job_admin.', 'job_blue-collar', 'job_entrepreneur',
       'job_housemaid', 'job_management', 'job_retired', 'job_self-employed',
       'job_services', 'job_student', 'job_technician', 'job_unemployed',
       'job_unknown', 'marital_divorced', 'marital_married', 'marital_single',
       'education_primary', 'education_secondary', 'education_tertiary',
       'education_unknown', 'default_no', 'default_yes', 'housing_no',
       'housing_yes', 'loan_no', 'loan_yes', 'contact_cellular',
       'contact_telephone', 'contact_unknown', 'month_apr', 'month_aug',
       'month_dec', 'month_feb', 'month_jan', 'month_jul', 'month_jun',
       'month_mar', 'month_may', 'month_nov', 'month_oct', 'month_sep',
       'poutcome_failure', 'poutcome_other', 'poutcome_success',
       'poutcome_unknown']]

features=bank.columns[1:]
target=bank.columns[0]

#Standardizing Each Column Values
standard_bank = pd.DataFrame(data = scale(bank[features]), columns = features)
standard_bank['Subscribe']=bank['Subscribe']


train_data,test_data = train_test_split(standard_bank,test_size=0.3,random_state=8)

#Model Building

logistic_regressor = LogisticRegression(penalty='l2')
logistic_regressor.fit(train_data[features],np.ravel(train_data[target]))
#logistic_regressor.predict_proba (train_data[features])
train_pred = logistic_regressor.predict(train_data[features])
test_pred = logistic_regressor.predict(test_data[features])

print("Confusion Matrix for Test Data: ",confusion_matrix(test_data[target],test_pred))
print("Confusion Matrix for Train Data: ",confusion_matrix(train_data[target],train_pred))

#Classification report

print('Classification Report on TrainData: ',classification_report(train_data[target],train_pred))
#Because of fewer number of records available for 'yes' class, model finds hard to Classify record to 'yes' class correctly

# Train Accuracy
print("Training Accuracy achieved by Logistic Regression(%): ",100*(sum(train_data[target]==train_pred)/train_data.shape[0]))

#Test Accuracy
print("Testing Accuracy achieved by Logistic Regression(%): ",100*(sum(test_data[target]==test_pred)/test_data.shape[0]))