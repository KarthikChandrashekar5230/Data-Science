import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

fire = pd.read_csv("C:\\Users\\kp\\Pictures\\Assignments\\Support Vector Machines\\forestfires.csv",header=0)

# Data Preprocessing prior to Model Building

fire["FireRange"] = 0
fire.loc[fire.size_category == 'large', "FireRange"] = 1
fire.drop(['month', 'day','size_category'], axis=1, inplace=True)
features=fire.columns[0:28]
target=fire.columns[28]

# Standardizing Each Column Values
standard_fire = pd.DataFrame(data=scale(fire[features]), columns=features)
standard_fire[target] = fire[target]


train_data,test_data = train_test_split(standard_fire,test_size=0.2,random_state=8)

#Model Building

model_svm=SVC(C=3.0,kernel = "linear",class_weight='balanced',probability=True,random_state=8)
model_svm.fit(train_data[features],train_data[target])
train_preds = model_svm.predict(train_data[features])
test_preds= model_svm.predict(test_data[features])
train_proba=model_svm.predict_proba(train_data[features])

# confusion matrix using pandas CrossTab
print("Confusion Matrix: \n",confusion_matrix(train_data[target],train_preds))

#Classification Report on Model's Performance for individual Classes
print("Classification Report of SVM Model Without Kernel: \n",classification_report(train_data[target],train_preds))

# Train Accuracy
print("Accuracy achieved on Training Dataset by Support Vector Machines Without Kernel(%): ",100*(np.mean(train_data[target]==train_preds)))

# Test Accuracy
print("Accuracy achieved on Test Dataset by Support Vector Machines With Linear Kernel(%): ",100*(np.mean(test_data[target]==test_preds)))


# ROC curve to determine Model's performance and the appropriate Threshold Level Value
# fpr => false positive rate
# tpr => true positive rate

fpr, tpr, thresholds = metrics.roc_curve(train_data.FireRange,train_proba[:,1]) #applicable for binary classification Problem
roc_auc = metrics.auc(fpr, tpr) # Area under ROC curve

plt.figure(figsize=(18,10))
plt.plot(fpr,tpr,color='darkorange',label='ROC curve (area = %0.4f)' % roc_auc)
plt.xlabel("False Positive Rate[FPR]")
plt.ylabel("True Positive Rate[TPR]")
plt.title('Receiver Operating Characteristic Curve[ROC]')
plt.legend(loc="lower right")
plt.show()

print("Model's Performance using ROC Curve(%): ",(roc_auc)*100)