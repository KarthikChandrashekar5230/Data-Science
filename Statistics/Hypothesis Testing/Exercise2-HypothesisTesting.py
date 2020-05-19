import pandas as pd
from scipy import stats

lab = pd.read_csv("C:\\Users\\kp\\Pictures\\Assignments\\Hypothesis Testing\\LabTAT.csv", header=0)

# Normality test to determine whether the Sample Set takes Normal/Gaussian distribution or not
distribution1 = stats.shapiro(lab['Laboratory 1'])
distribution2 = stats.shapiro(lab['Laboratory 2'])
distribution3 = stats.shapiro(lab['Laboratory 3'])
distribution4 = stats.shapiro(lab['Laboratory 4'])

print('Normality Test:')
if (distribution1[1]  > 0.05 and distribution2[1]  > 0.05 and distribution3[1]  > 0.05 and distribution4[1]  > 0.05):
    print("Data from all the Laboratories follows Normal Distribution")

else:
    print("Data from one or more Laboratories doesn't follow Normal Distribution")

# Hypotheses Definition
# H0: There is no significant difference/Varaition between the sample sets i.e two or more groups have same population mean
# H1: There is an effective difference between two or more Sample sets i.e two or more groups have diferent population mean
#Significance is choosen to be 5%

#One Way Anova Test selected for this kind of hypothesis Testing which involves in determining the variance between the Sample Sets
f_stats,pvalueF=stats.f_oneway(lab['Laboratory 1'],lab['Laboratory 2'],lab['Laboratory 3'],lab['Laboratory 4'])
print("P-Value of F-Score:",pvalueF)

if (pvalueF < 0.05):
    print("Reject the Null Hypothesis: There is an effective difference between two or more Sample sets by Anova Test")

else:
    print("Accept/Fail to Reject the Null Hypothesis: There is no significant difference/Varaition between the sample sets i.e two or more groups have same population mean by Anova Test")
