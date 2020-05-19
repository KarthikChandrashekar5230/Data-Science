import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

bank = pd.read_csv("C:\\Users\\kp\\Pictures\\Assignments\\Logistic Regression\\bank-full.csv", header=0, delimiter=';')


def data_visualization(bank):
    # Data Distribution - Boxplot of continuous variables wrt to Categorical Columns

    sb.boxplot(x="y", y="age", data=bank, palette="hls")
    plt.show()
    # There seems to be not much variation of age wrt to target Variable Values(y) as the median for both cases lie in somewhat same range

    sb.boxplot(x="y", y="balance", data=bank, palette="hls")
    plt.show()
    # Average Balance maintained by individual for Subscribing and Not Subscribing case is to be very low for most of the Individuals as the Median for both
    # the cases lies somewhat towards Zero Value from the plot

    sb.boxplot(x="y", y="duration", data=bank, palette="hls")
    plt.show()
    # Its Evident from the plot that people who spend more time in the Calls talking to the Executive have higher chances of getting
    # Subscribed to the term Deposit

    sb.boxplot(x="y", y="previous", data=bank, palette="hls")
    plt.show()
    # Cannot draw inferences from this Plot of Target Vs Previous as most of the records are highly concentrated close to Zero Value
    # for both Values of Target Variable

    sb.boxplot(x="marital", y="duration", data=bank, palette="hls")
    plt.show()
    # From the plot it seems to be that Married People Spend more time talking to the Executive rather than Single or Divorced people

    sb.boxplot(x="job", y="duration", data=bank, palette="hls")
    plt.show()
    # Its evident from the plot that Different Job type people spending time talking to Executive would differ which means the Job
    # category and Duration Spent are not independent variables

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

features=['balance', 'duration', 'campaign', 'job_admin.', 'job_management', 'job_retired',
       'job_student', 'job_technician', 'job_unemployed', 'marital_married',
       'education_tertiary','housing_no', 'loan_no', 'contact_cellular',
       'contact_telephone', 'month_apr', 'month_aug',
       'month_feb', 'month_jan', 'month_jul', 'month_jun',
       'month_mar', 'month_may', 'month_nov', 'poutcome_other', 'poutcome_success']
target=['Subscribe']

#Standardizing Each Column Values
standard_bank = pd.DataFrame(data = scale(bank[features]), columns = features)
standard_bank['Subscribe']=bank['Subscribe']


train_data,test_data = train_test_split(standard_bank,test_size=0.3,random_state=8)

#Model Building
# Building a model on train data set

logistic_regressor = sm.Logit(train_data[target],sm.add_constant(train_data[features])).fit_regularized(method='l1',maxiter=100)

#summary
print(logistic_regressor.summary())

#pd.set_option('mode.chained_assignment', None)
train_preds_prob = logistic_regressor.predict(sm.add_constant(train_data[features]))

# Creating new column for storing predicted class of Target Variable and filling all the cells with zeros in that Column initially
train_data=train_data.assign(Predictions=0)

# Taking threshold value as 0.5 and the probability value above 0.5 will be treated as correct value
train_data.loc[train_preds_prob >= 0.20,"Predictions"] = 1

# confusion matrix using pandas CrossTab
confusion_matrix = pd.crosstab(train_data["Subscribe"],train_data["Predictions"],margins=True)
print("Confusion Matrix: ", confusion_matrix)

print("Classification Report of the Logit Model: ",classification_report(train_data["Subscribe"],train_data["Predictions"]))

# ROC curve to determine Model's performance and the appropriate Threshold Level Value
# fpr => false positive rate
# tpr => true positive rate

fpr, tpr, thresholds = metrics.roc_curve(train_data["Subscribe"],train_preds_prob) #applicable for binary classification Problem
roc_auc = metrics.auc(fpr, tpr) # Area under ROC curve

plt.figure(figsize=(15,10))
plt.plot(fpr,tpr,color='darkorange',label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlabel("False Positive Rate[FPR]")
plt.ylabel("True Positive Rate[TPR]")
plt.title('Receiver Operating Characteristic Curve[ROC]')
plt.legend(loc="lower right")
plt.show()

print("Model's Performance using ROC Curve on Train DataSet: ",roc_auc)

# Train Accuracy
print("Training Accuracy achieved by StatsModels Logistic Regression(%): ",100*(sum(train_data["Subscribe"]==train_data["Predictions"])/train_data.shape[0]))


# Prediction on Test data set

test_preds_prob = logistic_regressor.predict(sm.add_constant(test_data[features]))
test_data=test_data.assign(Predictions=0)
test_data.loc[test_preds_prob >= 0.20,"Predictions"] = 1

# confusion matrix using pandas CrossTab
confusion_matrix = pd.crosstab(test_data["Subscribe"],test_data["Predictions"],margins=True)

#Test Accuracy
print("Testing Accuracy achieved by StatsModels Logistic Regression(%): ",100*(sum(test_data["Subscribe"]==test_data["Predictions"])/test_data.shape[0]))