import pandas as pd
from scipy import stats
import numpy as np

sales = pd.read_csv("C:\\Users\\kp\\Pictures\\Assignments\\Hypothesis Testing\\BuyerRatio.csv", header=0)
array_sales=np.array(sales.iloc[:,1:])

# Hypotheses Definition
# H0: Gender Buyer ratio and Specified Regions are independent.
# H1: Gender Buyer ratio and Specified Regions are not independent
#Significance Level aplha set to 5%

#Chi-Square Test is selected for this problem as the Data is of Categorical type
Chisquares_results=stats.chi2_contingency(array_sales)
Chi_pValue=Chisquares_results[1]
print("P-Value from Chi-Square Distribution is: ",Chi_pValue)

if (Chi_pValue < 0.05):
    print("Reject the Null Hypothesis: Gender Buyer ratio and Specified Regions are not independent by Chi-Square test")

else:
    print("Accept/Fail to Reject the Null Hypothesis: Gender Buyer ratio and Specified Regions are independent by Chi-Square test")