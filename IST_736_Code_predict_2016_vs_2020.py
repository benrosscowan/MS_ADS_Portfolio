#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Importing Libraries
import os
import pandas as pd
import re
import string
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import plot_confusion_matrix


import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score

import emoji


# In[113]:


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)


# In[2]:


oldDF = pd.read_csv('C:/Users/katva/Desktop/old_tweets.csv',nrows=(400))
oldDF.head

def cleaner(tweet):
    tweet = re.sub("@[A-Za-z0-9]+","",str(tweet)) #Remove @ sign
    tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", str(tweet)) #Remove http links
    tweet = " ".join(tweet.split())
    tweet = ''.join(c for c in tweet if c not in emoji.UNICODE_EMOJI) #Remove Emojis
    tweet = tweet.replace("#", "").replace("_", " ") #Remove hashtag sign but keep the text
    return tweet
oldDF['text'] = oldDF['text'].apply(lambda x: cleaner(x))
oldDF['text'].head()
oldDF.shape


# In[3]:


#https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python
def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)
oldDF['text'] = oldDF['text'].apply(lambda x: deEmojify(x))
oldDF.head()


# In[4]:


#### Now reading in newer tweets
newDF = pd.read_csv('C:/Users/katva/Desktop/new_tweets.csv')
newDF.head()
newDF.info()
oldDF.shape
oldDF.columns


# In[5]:


#run newDF throughcleaner and deEmojify
newDF['text'] = newDF['text'].apply(lambda x: cleaner(x))
#newDF['text'] = newDF['text'].apply(lambda x: deEmojify(x))
newDF.head()


# In[6]:


##now combine dataframes 
tweetsDF = pd.concat([oldDF, newDF], axis=0)


# In[7]:


## checking out the new DF
tweetsDF.columns
tweetsDF.head()
tweetsDF.tail()
tweetsDF.shape## there are 797 rows and 2 columns
tweetsDF.isna().sum()# 1 year is missing from the data


# In[8]:


#removing the row with the nan
tweetsDF = tweetsDF.dropna()
tweetsDF.head # this still leaves 796
tweetsDF.dtypes
tweetsDF.year = tweetsDF.year.astype(int)##convert the floats to int


# In[9]:


#Converting columns to lists for countvectorizer and classification analysis later
tweet_list = tweetsDF['text'].tolist()

year_list = tweetsDF['year'].tolist()

print(year_list)


# In[10]:


# Interface lemma tokenizer from nltk with sklearn Code from:https://gist.github.com/4OH4/f727af7dfc0e6bb0f26d2ea41d89ee55
class LemmaTokenizer:
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`']
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if t not in self.ignore_tokens]


# In[11]:


###########  CountVectorizer with stemming   #####
CV = CountVectorizer(input='content', 
                              analyzer='word',
                              stop_words='english',
                              #token_pattern='(?u)[a-zA-Z]+',
                              #token_pattern='pattern',
                              tokenizer=LemmaTokenizer(),
                              #strip_accents = 'unicode',
                              lowercase=True
                              )


# In[12]:


#Vectorize
vect=CV.fit_transform(tweet_list)
vocab1 = CV.vocabulary_
print(vocab1)


# In[13]:


## Make Count Vect lem dataframe 
DF = pd.DataFrame(vect.toarray(), columns=vocab1)                             
DF.describe
DF.head()


# In[14]:


## Add  Labels back on dataframes
DF.insert(loc=0, column='old_or_new', value=year_list)
DF.head()
DF['old_or_new']


# In[15]:


## Replace the NaN with 0 
DF=DF.fillna(0)


# In[16]:


##Split data into testing and training sets
TrainDF, TestDF = train_test_split(DF, test_size=0.3, random_state=0)


# In[17]:


## SEPERATE LABELS 
## Save labels
TestDFLabs=TestDF['old_or_new']
print(TestDFLabs)

## remove labels
TestDF = TestDF.drop(['old_or_new'], axis=1)
print(TestDF)

TrainDFLabs=TrainDF.drop(['old_or_new'], axis=1)
print(TrainDFLabs)
TrainLabs=TrainDF['old_or_new']
print(TrainLabs)


# In[19]:


######### Build NB Model for Count Vect Stemmed Data ##########
MyModelNB= MultinomialNB()

MyModelNB.fit(TrainDFLabs, TrainLabs)

y_pred = MyModelNB.predict(TestDF)
print("The prediction from NB is:")
print(y_pred)
print("The actual labels are:")
print(TestDFLabs)
# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score -> ", accuracy_score(y_pred, TestDFLabs)*100)

acc = accuracy_score(y_pred, TestDFLabs)

## confusion matrix
cmNB = confusion_matrix(TestDFLabs, y_pred)
print("The confusion matrix is:")
print(cmNB)

plt.figure(figsize=(9,9))
sns.heatmap(cmNB, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'viridis');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(acc)
plt.title(all_sample_title, size = 15);

print("Number of mislabeled points out of a total %d points : %d" % (TestDF.shape[0], (TestDFLabs != y_pred).sum()))


# In[20]:


######### Build SVM Model for Count Vect with stemming Data ##

# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto') #ran with kernel = 'linear' = 99% (overfitting?),'rbf'= 81%,'poly' =acc 46%
SVM.fit(TrainDFLabs, TrainLabs)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(TestDF)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, TestDFLabs)*100)

acc = accuracy_score(predictions_SVM, TestDFLabs)

# Confusion Matrix
SVM_matrix = confusion_matrix(TestDFLabs, predictions_SVM)
print("\nThe confusion matrix is:")
print(SVM_matrix)


plt.figure(figsize=(9,9))
sns.heatmap(SVM_matrix, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'viridis');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(acc)
plt.title(all_sample_title, size = 15);


# In[23]:


###### Looking at word freq btwn old and new

#Converting columns of both df to lists for countvectorizer 
oldDF.dropna()
old_list = oldDF['text'].tolist()

newDF.dropna()
new_list = newDF['text'].tolist()

## using CV from previous step
old_matrix = CV.fit_transform(old_list)
old_freqs = zip(CV.get_feature_names(), old_matrix.sum(axis=0).tolist()[0])    
# sort from largest to smallest
print (sorted(old_freqs, key=lambda x: -x[1]))


# In[22]:


new_matrix = CV.fit_transform(new_list)
new_freqs = zip(CV.get_feature_names(), new_matrix.sum(axis=0).tolist()[0])    
# sort from largest to smallest
print (sorted(new_freqs, key=lambda x: -x[1]))


# In[24]:


DF['old_or_new'].head()


# In[25]:


DF['old_or_new'].tail()


# In[139]:


# sampling section to attempt to balance the data a bit better
new = DF.loc[DF['old_or_new']==2020]
old = DF.loc[DF['old_or_new']==2016]


# In[142]:


new.shape ##396 rows of 3559 features


# In[144]:


old.shape ## 400 rows of 3559 features


# In[78]:


new_sample = new.sample(n=350,random_state = 2, replace = False) 
old_sample = old.sample(n=350,random_state = 2, replace = False)
## tried n=350 yielded NB accuracy of 94%, mislabled = 11 and increased to 96% accuracy after 5 runs of the model
## tried n=350 with random_state = 2 yielded NB accuracy of 96%, mislabled = 7
## tried n=350 with random_state = 3 yielded NB accuracy of 97%, mislabled = 5
## n = 200 yielded NB accuracy of 94%, mislabeled = 7
## n = 200 with random_state = 2 yielded NB accuracy of 95%, mislabeled = 5
## n = 200 with random_state = 3 yielded NB accuracy of 95%, mislabeled = 5
## n = 800 with replace = True yielded 99% with mislabled = 4 because of definite overfitting, done for comparison sake

df = new_sample.append([old_sample])


# In[79]:


##Split data into testing and training sets
TrainDF2, TestDF2 = train_test_split(df, test_size=0.3, random_state=0)


# In[80]:


## SEPERATE LABELS 
## Save labels
TestDFLabs2=TestDF2['old_or_new']
print(TestDFLabs2)

TrainLabs2=TrainDF2['old_or_new']
print(TrainLabs2)

## remove labels
TestDF2 = TestDF2.drop(['old_or_new'], axis=1)
print(TestDF2)

TrainDF2=TrainDF2.drop(['old_or_new'], axis=1)
print(TrainDF2)


# In[81]:


######### Build NB Model for Count Vect Stemmed Data ##########
MyModelNB= MultinomialNB()

MyModelNB.fit(TrainDF2, TrainLabs2)

y_pred2 = MyModelNB.predict(TestDF2)
print("The prediction from NB is:")
print(y_pred2)
print("The actual labels are:")
print(TestDFLabs2)
# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score -> ", accuracy_score(y_pred2, TestDFLabs2)*100)

acc2 = accuracy_score(y_pred2, TestDFLabs2)

## confusion matrix
cmNB2 = confusion_matrix(TestDFLabs2, y_pred2)
print("The confusion matrix is:")
print(cmNB2)

plt.figure(figsize=(9,9))
sns.heatmap(cmNB2, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'viridis');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(acc2)
plt.title(all_sample_title, size = 15);

print("Number of mislabeled points out of a total %d points : %d" % (TestDF2.shape[0], (TestDFLabs2 != y_pred2).sum()))


# In[89]:


######### Build SVM Model for Count Vect with stemming Data ##

# fit the training dataset on the classifier
SVM2 = svm.SVC(C=10.0, kernel='poly', degree=3, gamma='scale') 
#ran with kernel = 'linear' = 99% (overfitting?),'rbf'= 81%,
#'poly' with degree= 3 yielded acc 94% mislabled 12
#tried (C=1.0, kernel='linear', degree=3, gamma='auto') yielded accuracy of 98% mislabled 3, 
# changing c= made little to no change neither did changing gamma to 'scale'



SVM2.fit(TrainDF2, TrainLabs2)
# predict the labels on validation dataset
predictions_SVM2 = SVM2.predict(TestDF2)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM2, TestDFLabs2)*100)

acc2 = accuracy_score(predictions_SVM2, TestDFLabs2)

# Confusion Matrix
SVM_matrix2 = confusion_matrix(TestDFLabs2, predictions_SVM2)
print("\nThe confusion matrix is:")
print(SVM_matrix2)


plt.figure(figsize=(9,9))
sns.heatmap(SVM_matrix2, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'viridis');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(acc2)
plt.title(all_sample_title, size = 15);

print("Number of mislabeled points out of a total %d points : %d" % (TestDF2.shape[0], (TestDFLabs2 != predictions_SVM2).sum()))


# In[90]:


## sentiment differences
import textblob
from textblob import TextBlob


# In[91]:


y = oldDF['year']
X = oldDF['text']


# In[92]:


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(oldDF.text, oldDF.year, test_size=0.2,random_state=109) # 80% training and 20% test


# In[93]:


from nltk.stem import WordNetLemmatizer
lemmatiser = WordNetLemmatizer()
# Defining a module for Text Processing
def text_process(tex):
    # 1. Removal of Punctuation Marks 
    nopunct=[char for char in tex if char not in string.punctuation]
    nopunct=''.join(nopunct)
    # 2. Lemmatisation 
    a=''
    i=0
    for i in range(len(nopunct.split())):
        b=lemmatiser.lemmatize(nopunct.split()[i], pos="v")
        a=a+b+' '
    # 3. Removal of Stopwords
    return [word for word in a.split() if word.lower() not 
            in stopwords.words('english')]


# In[94]:


# defining the bag-of-words transformer on the text-processed corpus # i.e., text_process() 
bow_transformer=CountVectorizer(analyzer=text_process).fit(X_train)
# transforming into Bag-of-Words and hence textual data to numeric..
text_bow_train=bow_transformer.transform(X_train)#ONLY TRAINING DATA
# transforming into Bag-of-Words and hence textual data to numeric..
text_bow_test=bow_transformer.transform(X_test)#TEST DATA


# In[95]:


# transforming into Bag-of-Words and hence textual data to numeric..
text_bow_train=bow_transformer.transform(X_train)#ONLY TRAINING DATA
text_bow_train.shape


# In[96]:


# transforming into Bag-of-Words and hence textual data to numeric..
text_bow_test=bow_transformer.transform(X_test)#TEST DATA
text_bow_test.shape


# In[97]:


# old DF
oldDF['polarity'] = oldDF['text'].apply(lambda x: TextBlob(x).polarity)
oldDF['subjective'] = oldDF['text'].apply(lambda x: TextBlob(x).subjectivity)


# In[98]:


#polarity ranges from -1 to 1, with -1 being negative and 1 being positive.
oldDF['polarity']


# In[99]:


avgOLDpol = sum(oldDF['polarity'])/400 #400 is the length of the doc
print(avgOLDpol)


# In[134]:


pos_old = oldDF.loc[oldDF['polarity']>0.0]


# In[135]:


print(pos_old)


# In[137]:


pos_old.shape ## 140 positive


# In[133]:


neg_old = oldDF.loc[oldDF['polarity']<0.0]
print(neg_old.head)


# In[136]:


neg_old.shape # 339 rows of neg


# In[100]:


#subjectivity ranges from 0 to 1, with 0 being objective and 1 being subjective.
oldDF['subjective']


# In[101]:


avgOLDsubj = sum(oldDF['subjective'])/400
print(avgOLDsubj)


# In[122]:


subj_old = oldDF.loc[oldDF['subjective']<0.5]
print(subj_old.head(10))


# In[132]:


obj_old = oldDF.loc[oldDF['subjective']<0.5]
print(obj_old)


# In[102]:


## New DF
newDF['polarity'] = newDF['text'].apply(lambda x: TextBlob(x).polarity)
newDF['subjective'] = newDF['text'].apply(lambda x: TextBlob(x).subjectivity)


# In[103]:


#polarity ranges from -1 to 1, with -1 being negative and 1 being positive.
newDF['polarity']


# In[104]:


avgNEWpol = sum(newDF['polarity'])/397 #397 is the length of the doc
print(avgNEWpol)


# In[130]:


pos_new = newDF.loc[newDF['polarity']>0.0]
pos_new.shape
print(pos_new.head) 


# In[131]:


neg_new = newDF.loc[newDF['polarity']<0.0]
neg_new.shape
print(neg_new.head)


# In[105]:


#subjectivity ranges from 0 to 1, with 0 being objective and 1 being subjective.
newDF['subjective']


# In[107]:


avgNEWsubj = sum(newDF['subjective'])/397
print(avgNEWsubj)


# In[119]:


subj_new = newDF.loc[newDF['subjective']>0.5]
print(subj_new.head(10))


# In[121]:


obj_new = newDF.loc[newDF['subjective']<0.5]
print(obj_new.head(10))

