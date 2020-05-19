import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from collections import Counter
import matplotlib.pyplot as plt
import re

database = []
# Add the Contents of csv to the list such that Each Transaction is Separarted by \n
with open("C:\\Users\\kp\\Pictures\\Assignments\\Association Rule Mining[ARM]\\groceries.csv") as f:
    database = f.read()

# splitting the data into separate transactions using separator as "\n", converting a single string into list of strings
database = database.split("\n")
transactions_list = []
# Converting list of strings to list of list of strings
for i in database:
    transactions_list.append(i.split(","))

all_items_database = [i for item in transactions_list for i in item]

item_frequencies = Counter(all_items_database)
item_frequencies = sorted(item_frequencies.items(), key=lambda x: x[1])

# Storing frequencies and items in separate variables
frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))

#Bar plot of Items frequency
plt.figure(figsize=(20, 15))
plt.bar(height=frequencies[0:11], x=items[0:11], color='rgbkymc')
plt.xlabel("Items")
plt.ylabel("Frequency of Items in Transactional Database")

# Purpose of converting all lists into Series objects which is further converted to Dataframe
transaction_groceries  = pd.DataFrame(pd.Series(transactions_list))
transaction_groceries = transaction_groceries.iloc[:9835, :] # removing the last empty transaction
transaction_groceries.columns = ["Transactions"]
# creating a dummy columns for the each item in each transactions ... Using column names as item name
X = transaction_groceries['Transactions'].str.join(sep='*').str.get_dummies(sep='*')
frequent_itemsets = apriori(X, min_support=0.005, max_len=3,use_colnames = True)
frequent_itemsets.sort_values('support',ascending = False,inplace=True)
#Convert Frozen sets to list of Lists
dummy_variable=[list(x) for x in frequent_itemsets.itemsets[0:11]]
#Convert List of Lists to List
frequency_items_list=[item for sublist in dummy_variable for item in sublist]

#Bar plot of Itemsets with the corresponding support
plt.figure(figsize=(20,15))
plt.bar(x=frequency_items_list,height = frequent_itemsets.support[0:11],color='rgmyk')
plt.xlabel('Item-Sets')
plt.ylabel('Support')

rules_mining = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

#Convert Frozen set to list
def to_list(i):
    return (list(i))

redundant_list = rules_mining.antecedents.apply(to_list) + rules_mining.consequents.apply(to_list)

#Sort the list such that the elements in the list are arranged in alphabetical order
redundant_list = redundant_list.apply(sorted)

#Convert list of strings to string
redundant_list=redundant_list.str.join(',')
rules_mining["Antecedents_Consequents"]=pd.Series(redundant_list)

#Removing Rules that are repeated more than once which has the same support which is not required
rules_mining.drop_duplicates(subset ="Antecedents_Consequents", keep ='first', inplace = True)
rules_mining.sort_values('lift', ascending=False).head(10)