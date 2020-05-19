import pandas as pd
from scipy import stats

cutlet = pd.read_csv("C:\\Users\\kp\\Pictures\\Assignments\\Hypothesis Testing\\Cutlets.csv", header=0)

# Normality test to determine whether the Sample Set takes Normal/Gaussian distribution or not
stats_scoreA, pvalueA = stats.shapiro(cutlet['Unit A'])
stats_scoreB, pvalueB = stats.shapiro(cutlet['Unit B'])
if (pvalueA > 0.05 and pvalueB > 0.05):
    print("Unit A and Unit B data follows Normal Distribution")

else:
    print("Unit A and Unit B data doesn't follow Normal Distribution")

# Hypotheses Definition
# H0: Mean Diameter of Cutlet in Unit A = Mean Diameter of Cutlet in Unit B
# H1: Mean Diameter of Cutlet in Unit A != Mean Diameter of Cutlet in Unit B

# Variance Test
var_stat, pvalue_var = stats.levene(cutlet['Unit A'], cutlet['Unit B'])

if (pvalue_var < 0.05):
    print("Reject the Null Hypothesis: There is a Significant Difference in Sample Units by Variance Test")

else:
    print(
        "Accept/Fail to Reject the Null Hypothesis: There is no difference between Sample Units i.e. Sample Units have equal Variance by Variance Test")

# Two Tailed 2-Sample t-test Statistics is used
ttest_stats, pvalue_ttest = stats.ttest_ind(cutlet['Unit A'], cutlet['Unit B'])

if (pvalue_var < 0.05):
    print("Reject the Null Hypothesis: There is a Significant Difference in Sample Units by t-test")

else:
    print(
        "Accept/Fail to Reject the Null Hypothesis: There is no difference between Sample Units i.e. Sample Units have equal Variance by t-test")

