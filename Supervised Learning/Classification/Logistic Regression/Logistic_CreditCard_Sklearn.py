import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import statsmodels.formula.api as sm
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

card = pd.read_csv("C:\\Users\\kp\\Pictures\\Assignments\\Logistic Regression\\creditcard.csv", header=0)


def data_visualization(card):
    # Bar Plots for Categorical Columns

    sb.countplot(x="card", data=card,
                 palette="hls")  # Imbalanced Dataset as the values in Target Variable is unevenly distributed
    plt.show()
    sb.countplot(x="owner", data=card, palette="hls")  # Values for this attribute distributed somewhat uniformly
    plt.show()
    sb.countplot(x="selfemp", data=card, palette="hls")  # Values for this attribute distributed unevenly in the Dataset
    plt.show()

    pd.crosstab(card.card, card.owner).plot(kind="bar")  # Values for these attributes distributed somewhat uniformly
    plt.show()
    pd.crosstab(card.card, card.selfemp).plot(kind="bar")
    plt.show()
    # Theres a accountable variation between these attributes.Hence Acceptance of card also depends on the Selfemp attribute along with other parameters

    # Data Distribution - Boxplot of continuous variables wrt to categorical columns

    sb.boxplot(x="card", y="income", data=card, palette="hls")
    plt.show()
    # Theres not much difference between the median of the income for Accepted and Rejected Applications.So its quite hard to predict
    # Acceptance based on just income.

    sb.boxplot(x="card", y="expenditure", data=card, palette="hls")
    plt.show()
    # Its evident from the plot that applications will be rejected with smaller values of Expenditure

    sb.boxplot(x="card", y="age", data=card, palette="hls")
    plt.show()
    # Its evident from the plot that the Acceptance/Rejection of Application is less dependent on the age of the individual as the
    # Median for both Acceptance and Rejection lie in the same range or theres not noticable Margin between Acceptance and Rejection median

    sb.boxplot(x="owner", y="income", data=card, palette="hls")
    plt.show()
    # Its evident from the plot that people owning Home are tend to have High income than the people who don't own a Home

    return None


#data_visualization(card)


#Finding NaN Values in the Dataframe and eliminating them

#nan_records = card[card['income'].isnull()]               # To find records with NaN Values in it
card.dropna(inplace=True)
#There were no NaN values presnt in the Dataframe so the shape of Df remains same even after dropna() has been called

card["Acceptance"] = 0
card.loc[card.card == "yes","Acceptance"] = 1

dummies=pd.get_dummies(card[['owner','selfemp']])
card.drop(['Unnamed: 0','owner','selfemp','card'],axis=1,inplace=True)
card = pd.concat([card,dummies],axis=1)
card=card[['reports', 'age', 'income', 'share', 'expenditure', 'dependents','months', 'majorcards', 'active', 'owner_no', 'owner_yes','selfemp_no', 'selfemp_yes','Acceptance']]

#Standardizing Each Column Values
standard_card = pd.DataFrame(data = scale(card.iloc[:,0:13]), columns = ['reports', 'age', 'income', 'share', 'expenditure', 'dependents','months', 'majorcards', 'active', 'owner_no', 'owner_yes','selfemp_no','selfemp_yes'])
standard_card['Acceptance']=card['Acceptance']


train_data,test_data = train_test_split(standard_card,test_size=0.2,random_state=8)

features=['reports', 'age', 'income', 'share', 'expenditure', 'dependents','months', 'majorcards', 'active', 'owner_no', 'owner_yes', 'selfemp_no','selfemp_yes']
target=['Acceptance']


#Model Building

logistic_regressor = LogisticRegression(penalty='l2')
logistic_regressor.fit(train_data[features],np.ravel(train_data[target]))
#logistic_regressor.predict_proba (train_data[features])
train_pred = logistic_regressor.predict(train_data[features])
test_pred = logistic_regressor.predict(test_data[features])

print("Confusion Matrix for Test Data: ",confusion_matrix(test_data['Acceptance'],test_pred))
print("Confusion Matrix for Train Data: ",confusion_matrix(train_data['Acceptance'],train_pred))


# Train Accuracy
print("Training Accuracy achieved by Logistic Regression(%): ",100*(sum(train_data['Acceptance']==train_pred)/train_data.shape[0]))

#Test Accuracy
print("Testing Accuracy achieved by Logistic Regression(%): ",100*(sum(test_data['Acceptance']==test_pred)/test_data.shape[0]))


predictors = ['reports', 'income','share', 'dependents', 'active']
output=['Acceptance']
#Removed the features which are insignificant with the help of logit Summary based on p-values

#Alternate Model Building using StatsModels.formula.api
#No Need to add Constant Column Explicitly for the intercept in case of Statsmodels.formula.api.logit

logistic_classifier = sm.logit('Acceptance~reports+income+share+dependents+active',
                               data = card).fit_regularized(method='l1',maxiter=150)

#summary
print(logistic_classifier.summary())

preds_prob = logistic_classifier.predict(card[predictors])

# Creating new column for storing predicted class of Target Variable and filling all the cells with zeros in that Column initially
card=card.assign(Predictions=0)

# Taking threshold value as 0.5 and the probability value above 0.5 will be treated as correct value
card.loc[preds_prob >= 0.5,"Predictions"] = 1

# confusion matrix using pandas CrossTab
confusion_matrix = pd.crosstab(card["Acceptance"],card["Predictions"],margins=True)
print("Confusion Matrix: \n",confusion_matrix)

print("Classification Report of the Logit Model: \n",classification_report(card["Acceptance"],card["Predictions"]))


# ROC curve to determine Model's performance and the appropriate Threshold Level Value
# fpr => false positive rate
# tpr => true positive rate

fpr, tpr, thresholds = metrics.roc_curve(card["Acceptance"],preds_prob) #applicable for binary classification Problem
roc_auc = metrics.auc(fpr, tpr) # Area under ROC curve

plt.figure(figsize=(15,10))
plt.plot(fpr,tpr,color='darkorange',label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlabel("False Positive Rate[FPR]")
plt.ylabel("True Positive Rate[TPR]")
plt.title('Receiver Operating Characteristic Curve[ROC]')
plt.legend(loc="lower right")
plt.show()

print("Model's Performance using ROC Curve: ",roc_auc)

# Accuracy Calculations
print("Accuracy achieved by StatsModels Logistic Regression(%): ",100*(sum(card["Acceptance"]==card["Predictions"])/card.shape[0]))