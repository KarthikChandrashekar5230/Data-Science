import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

raw_df = pd.read_csv('C:\\Users\\kp\\Pictures\\Assignments\\Naive Bayes\\sms_raw_NB.csv',header=0,encoding='latin-1')
raw_df.columns=['Type','Text']

raw_df['Text'] = raw_df['Text'].astype('str')
raw_df['Type'] = raw_df['Type'].astype('str')

raw_df = raw_df.fillna(0)
raw_df.drop(raw_df[raw_df['Text'] == 0].index, axis=0, inplace=True)
raw_df.drop(raw_df[raw_df['Type'] == 0].index, axis=0, inplace=True)
raw_df.drop_duplicates(subset='Text', keep='first', inplace=True)

raw_df = raw_df.reset_index(drop=True)

raw_df.Text = raw_df.Text.str.lower()

raw_df['Text'] = raw_df['Text'].str.replace("'s'", "")

raw_df["Text"] = raw_df['Text'].apply(lambda record: word_tokenize(record))
stop_words = set(stopwords.words("english"))
raw_df['Text'] = raw_df['Text'].apply(lambda record: [word for word in record if word not in stop_words])


def apply_lemmatization(string_list):
    lem = WordNetLemmatizer()
    list = []

    for word in string_list:
        list.append(lem.lemmatize(word, "v"))

    return list


raw_df['Text'] = raw_df['Text'].apply(apply_lemmatization)

raw_df.Text = raw_df.Text.apply(lambda record: " ".join(record))

raw_df["Text"] = raw_df['Text'].apply(lambda x: re.sub('[^A-Za-z" "]+', "", x))

class_codes = {'ham': 1, 'spam': -1}
raw_df['Class_Code'] = raw_df['Type']
raw_df = raw_df.replace({'Class_Code': class_codes})
cleaned_df = raw_df

X_train, X_test, y_train, y_test = train_test_split(cleaned_df['Text'],
                                                    cleaned_df['Class_Code'], test_size=0.15,random_state=8)

tfidf = TfidfVectorizer(encoding='utf-8',
                                ngram_range=(1, 2),
                                stop_words=None,
                                lowercase=False,
                                max_df=1.0,
                                min_df=10,
                                max_features=700,
                                norm='l2',
                                sublinear_tf=True)
features_train = tfidf.fit_transform(X_train).toarray()
class_labels_train = y_train

features_test = tfidf.transform(X_test).toarray()
class_labels_test = y_test


gaussian_nbc = GaussianNB()
pred_gaussian = gaussian_nbc.fit(features_train,class_labels_train).predict(features_test)
pred_gaussian_train=gaussian_nbc.predict(features_train)
confusion_matrix(y_test,pred_gaussian)
print ("Test Accuracy of Gaussian Naive Bayes model accuracy(in %):", accuracy_score(y_test, pred_gaussian)*100)
print ("Train Accuracy of Gaussian Naive Bayes model accuracy(in %):", accuracy_score(y_train, pred_gaussian_train)*100)


multinomial_nbc = MultinomialNB()
pred_multinomial = multinomial_nbc.fit(features_train,class_labels_train).predict(features_test)
pred_multinomial_train=multinomial_nbc.predict(features_train)
print(confusion_matrix(y_test,pred_multinomial))
print ("Test Accuracy of MultiNomial Naive Bayes model accuracy(in %):", accuracy_score(y_test, pred_multinomial)*100)
print ("Train Accuracy of MultiNomial Naive Bayes model accuracy(in %):", accuracy_score(y_train, pred_multinomial_train)*100)
#MultiNomial Naive Bayes Classifier is prebable Classifier when the features Vectors are Documents or for Documents Classificatio Problem