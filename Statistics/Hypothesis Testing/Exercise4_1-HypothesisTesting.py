import pandas as pd
from scipy import stats
import numpy as np

fantaloons = pd.read_csv("C:\\Users\\kp\\Pictures\\Assignments\\Hypothesis Testing\\Faltoons.csv", header=0)
lists_list_fantaloons=[list(np.array(fantaloons[col].value_counts())) for col in fantaloons.columns]
lists_list_fantaloons=np.column_stack(lists_list_fantaloons)
print(lists_list_fantaloons)

# Hypotheses Definition
# H0: Gender ratio and Day of the week are independent
# H1: Gender ratio and Day of the week are not independent
#Significance Level aplha set to 5%

#Chi-Square Test is selected for this problem as the Data is of Categorical type
Chisquares_results=stats.chi2_contingency(lists_list_fantaloons)
Chi_pValue=Chisquares_results[1]
print("P-Value from Chi-Square Distribution is: ",Chi_pValue)

if (Chi_pValue < 0.05):
    print("Reject the Null Hypothesis: Gender ratio and Day of the week are not independent by Chi-Square test")

else:
    print("Accept/Fail to Reject the Null Hypothesis: Gender ratio and Day of the week are independent by Chi-Square test")

