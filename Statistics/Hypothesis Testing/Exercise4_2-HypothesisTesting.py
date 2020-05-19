import pandas as pd
from scipy import stats
import numpy as np

customer_forms = pd.read_csv("C:\\Users\\kp\\Pictures\\Assignments\\Hypothesis Testing\\Costomer+OrderForm.csv", header=0)
lists_list_customer_forms=[list(np.array(customer_forms[col].value_counts())) for col in customer_forms.columns]
lists_list_customer_forms=np.column_stack(lists_list_customer_forms)
print(lists_list_customer_forms)

# Hypotheses Definition
# H0: Defective ratio and Centres are independent
# H1: Defective ratio and Centres are not independent
#Significance Level aplha set to 5%

#Chi-Square Test is selected for this problem as the Data is of Categorical type
Chisquares_results=stats.chi2_contingency(lists_list_customer_forms)
Chi_pValue=Chisquares_results[1]
print("P-Value from Chi-Square Distribution is: ",Chi_pValue)

if (Chi_pValue < 0.05):
    print("Reject the Null Hypothesis: Defective ratio and Centres are not independent by Chi-Square test")

else:
    print("Accept/Fail to Reject the Null Hypothesis: Defective ratio and Centres are independent by Chi-Square test")
