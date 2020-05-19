import os
import pandas as pd
import re
import numpy
from sklearn.metrics.pairwise import cosine_similarity

books = pd.read_csv("C:\\Users\\kp\\Pictures\\Assignments\\Recommendation system\\books.csv", encoding='latin1',
                    header=0)
books = books.drop(['Unnamed: 0', 'users[, 1]'], axis=1)
books.columns = ["Title", "Author", 'Publisher', 'Ratings']


def eliminate_missing_values(dataframe):
    dataframe.fillna('NA', inplace=True)
    dataframe.drop(dataframe[dataframe['Title'] == 'NA'].index, inplace=True)
    dataframe.drop(dataframe[dataframe['Author'] == 'NA'].index, inplace=True)
    dataframe.drop(dataframe[dataframe['Publisher'] == 'NA'].index, inplace=True)

    return dataframe


books = eliminate_missing_values(books)

kp = []

for record in books['Title']:
    if ('http:' in record):
        temp = re.split(';', record)
        http_list = [i for i in range(0, len(temp)) if "http" in temp[i]]
        start = http_list[0]
        end = http_list[-1]
        kp = kp + temp[start:end]
    else:
        pass
for record in books['Author']:
    if ('http:' in record):
        temp = re.split(';', record)
        http_list = [i for i in range(0, len(temp)) if "http" in temp[i]]
        start = http_list[0]
        end = http_list[-1]
        kp = kp + temp[start:end]
    else:
        pass
for record in books['Publisher']:
    if ('http:' in record):
        temp = re.split(';', record)
        http_list = [i for i in range(0, len(temp)) if "http" in temp[i]]
        start = http_list[0]
        end = http_list[-1]
        kp = kp + temp[start:end]
    else:
        pass
list_lists = []

kp = [substring for substring in kp if (('http:' not in substring) and (
            re.findall('^[0-9]', substring) not in [['0'], ['1'], ['2'], ['3'], ['4'], ['5'], ['6'], ['7'], ['8'],
                                                    ['9']]))]
#Smaller list Defined
# author=[];title=[];publisher=[]
#
# #Lists which consists only Author,only Title,Only Publisher respectively
# for i in range(0,len(kp)):
#     if (i==0 or i==1 or i==2):
#         if(i==0):
#             title.append(kp[i])
#         elif(i==1):
#             author.append(kp[i])
#         else:
#             publisher.append(kp[i])
#     elif (i%3==0):
#          title.append(kp[i])
#     elif ((i-1)%3==0):
#          author.append(kp[i])
#     elif ((i-2)%3==0):
#          publisher.append(kp[i])

# Remove unwanted entries from the list
indexes = [336, 337, 800, 801, 1168, 1169, 1476, 1477, 2624, 2625, 3220, 3221, 3222, 3223, 3224, 4218, 4219, 4898, 4899,
           4900, 4901, 4902, 4903,
           4907, 4908, 4909, 4910, 4911, 5890, 5891, 6117, 6118, 6398, 6399, 6409, 6410, 6600, 6601, 6602, 6603, 6604,
           6605, 6606, 6607, 6608, 6609,
           6610, 6611, 6612, 6613, 7958, 7959, 8269, 8270, 8355, 8356, 8840, 8841, 9100, 9101]

for index in sorted(indexes, reverse=True):
    del kp[index]

for i in range(0, len(kp), 3):
    list_lists.append(list(kp[i:(i + 3)]))

df = pd.DataFrame(list_lists, columns=["Title", "Author", 'Publisher'])
df['Ratings'] = 0


def remove_unwanted_records(record):
    if ('http:' in record):
        record = 'NA'
        return record
    else:
        return record


books['Title'] = books['Title'].apply(remove_unwanted_records)
books['Author'] = books['Author'].apply(remove_unwanted_records)
books['Publisher'] = books['Publisher'].apply(remove_unwanted_records)
books = eliminate_missing_values(books)


modified_books=pd.concat([books,df],ignore_index=True)
modified_books['Author']=modified_books['Author'].str.lower()
modified_books['Publisher']=modified_books['Publisher'].str.lower()
modified_books.drop_duplicates(subset ="Title",keep = 'first', inplace = True)

vectorizer_df=pd.get_dummies(data=modified_books, columns=['Author', 'Publisher'])
vectorizer_df=vectorizer_df.drop(['Title'],axis=1)
vectorizer_df = vectorizer_df.astype(numpy.float32)
vector_matrix=vectorizer_df.to_numpy()

# Compute the Cosine Similarity matrix based on the vector_matrix
cosine_sim = cosine_similarity(vector_matrix, vector_matrix)

# Construct reverse mapping
indices = pd.Series(modified_books.index, index=modified_books['Title'])

# Function that takes in book title as input and outputs most similar books based on the parameters assumed
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all books with that book
    sim_scores = list(enumerate(cosine_sim[idx]))
    #Enumerate() method adds a counter to an iterable and returns it in a form of enumerate object

    # Sort the books based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the books of the 10 most similar books
    sim_scores = sim_scores[1:11]

    # Get the books indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar books
    return modified_books['Title'].iloc[movie_indices]

#Get Similar books to the given input based on the parameters taken into account
print(get_recommendations('And Then There Were None', cosine_sim))
