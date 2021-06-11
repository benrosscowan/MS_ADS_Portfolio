#IST 736 Final Project Code
#Members:
# Benjamin Cowan, Kathryn Egan, Silki Kaur, Brandon Smith

#Sentiment Analysis - MNB and SVM Classifier Predictions -
#LDA for Topic Modeling

#loading libraries

import requests
import json
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text 
import pandas as p
import numpy as np
import random as rd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

import nltk
nltk.download('vader_lexicon')
nltk.download('wordnet')

from nltk.stem import PorterStemmer
STEMMER=PorterStemmer()

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

##################################################################################################
#import and clean section
#choose the dataframe to analyze
df=p.read_csv("C:/Users/bcow1/OneDrive - Syracuse University/Text Mining/2016_tweets.txt", delimiter='\t', encoding='cp1252')

#or choose:
#df=p.read_csv("C:/Users/bcow1/OneDrive - Syracuse University/Text Mining/climate_twitter.txt", delimiter='\t')

new = []
for item in df['text'].values:
    Headline = item
    Headline = re.sub(r"http\S+", "", Headline)
    Headline=re.sub(r'[,.;@#?!&$\-\']+', ' ', Headline, flags=re.IGNORECASE)
    Headline=re.sub(r'\ +', ' ', Headline, flags=re.IGNORECASE)
    Headline=re.sub(r'\"', ' ', Headline, flags=re.IGNORECASE)
    Headline=re.sub(r'[^a-zA-Z]', " ", Headline, flags=re.VERBOSE)
    Headline=re.sub(r' s | ll | z | w | t | n ', ' ', Headline, flags=re.IGNORECASE)
    Headline=re.sub(r'\s+', ' ', Headline, flags=re.IGNORECASE)
    new.append(Headline)
df['clean_text'] = new  

##################################################################################################
#sentiment section

#create sentiment label
df['scores'] = df['clean_text'].apply(lambda clean_text: sia.polarity_scores(clean_text))
df['compound']  = df['scores'].apply(lambda score_dict: score_dict['compound'])
df['SentimentLabel'] = df['compound'].apply(lambda c: 0 if c < 0 else (1 if c == 0 else 2))

y = df['SentimentLabel'].values
#df.to_csv(".csv")    
unique, counts = np.unique(y, return_counts=True)
print("original\n", np.asarray((unique, counts)))
##################################################################################################
# sampling section

pos = df.loc[df['SentimentLabel']==2]
neu = df.loc[df['SentimentLabel']==1]
neg = df.loc[df['SentimentLabel']==0]
#print(pos.head())

pos_sample = pos.sample(n=800, random_state=3, replace = False)
neu_sample = neu.sample(n=800, random_state=3, replace = False)
neg_sample = neg.sample(n=800, random_state=3, replace = False)

df = pos_sample.append([neu_sample, neg_sample])

AllList = df['clean_text'].values
#create lists from the sample so LDA can find general topics in each
PositiveList = pos_sample['clean_text'].values
NeutralList = neu_sample['clean_text'].values
NegativeList = neg_sample['clean_text'].values

print(len(AllList))
print(len(PositiveList))
print(len(NegativeList))
print(len(NeutralList))

###################################################################################################
#start creating dataframe and labels

# make X and y needed for modeling
#print(df.head())
#clean_text = df['clean_text'].values
X = df['clean_text'].values
y = df['SentimentLabel'].values

unique, counts = np.unique(y, return_counts=True)
print("after sampling\n", np.asarray((unique, counts)))

print(len(X))
print(len(y))


##################################################################################################
#stemmer if needed

# use the porter stemmer
def MY_STEMMER(str_input):
    words = re.sub(r"[^A-Za-z\-]", " ", str_input).lower().split()
    words = [STEMMER.stem(w) for w in words]
    return words


##################################################################################################

stop_word = text.ENGLISH_STOP_WORDS.union(["feelthebern", "gogreen", "youtube", "pm", "mt", "keepitintheground", "fossilfuel", "auspol", "th", "congratsleo", "marcorubio", "berniesanders", "qu", "twitter", "adaption", "ab", "addressing", "affects", "affected", "hate", "abt", "fuck", "rt",  "mashablenews", "en", "di", "si", "need", "needed", "climateaction", "climatechangeisreal", "oscars","leo", "leonardo", "dicaprio", "https", "climatechange", "amp", "make", "ll", "adb", "leonardodicaprio", "leodicaprio", "https", "globalwarming", "change", "action", "changes", "li", "sn", "musk", "elon", "lo", "ol", "faa"])
     
#vectorizers
#tokenizer= MY_STEMMER,ngram_range=(2, 2),
MyCV=CountVectorizer(input="content", lowercase=True, min_df=2,  max_features=1000,stop_words = stop_word)
MyTfidfV = TfidfVectorizer(input='content',  use_idf=True, min_df=2, stop_words= stop_word, max_features = 1000, lowercase=True)

#sparse matrix
MyDTM_CV = MyCV.fit_transform(X)
MyDTM_Tfidf = MyTfidfV.fit_transform(X)

#list of vocabulary
vocab = MyCV.get_feature_names()
vocab2 = MyTfidfV.get_feature_names()

#create df
MyDTM_DF_CV = pd.DataFrame(MyDTM_CV.toarray(), columns = vocab)
MyDTM_DF_Tfidf = pd.DataFrame(MyDTM_Tfidf.toarray(), columns = vocab2)

#print(MyDTM_DF_Tfidf)

######################################################################################
#indicative words function 
def show_most_and_least_informative_features(vectorizer, clf, class_idx=0, n=10):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[class_idx], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[-n:])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))

######################################################################################

#determine cost and crossvalidation accuracies for tf and tfidf vectorized dataframes
X = MyDTM_DF_CV
#split dataframe for modeling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=2)

accuracy_list = []
for n in range(1,10):  
    #choose model:
    #model = SVC(C=n, kernel = 'linear')
    model = SVC(C=n, kernel = 'poly')
    #model = SVC(C=n, kernel = 'rbf') 
    
    model.fit(X_train, y_train)
    predict = model.predict(X_test)
    acc_score = accuracy_score(y_test, predict)
    accuracy_list.append((n,acc_score))   
# create a DF with the list
acc_score_df = pd.DataFrame(accuracy_list, columns=['C', 'Score'])
# print the DF by score
print(acc_score_df.sort_values('Score', ascending=False))

# term frequency vectorized dataframe - crossvalidation
model1 = MultinomialNB()
model2 = SVC(C=3, kernel = 'linear')
model3 = SVC(C=9, kernel = 'poly')
model4 = SVC(C=2, kernel = 'rbf')

print("tf - MNB:", np.average(cross_val_score(model1, X_train, y_train, cv=5)))
print("tf - SVC - linear:", np.average(cross_val_score(model2, X_train, y_train, cv=5)))
print("tf - SVC - poly:", np.average(cross_val_score(model3, X_train, y_train, cv=5)))
print("tf - SVC - rbf:", np.average(cross_val_score(model4, X_train, y_train, cv=5)))

#next,
#determine cost and crossvalidation accuracies for tfidf vectorized dataframes
X = MyDTM_DF_Tfidf
#split dataframe for modeling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=2)

accuracy_list = []
for n in range(1,10):    
    #choose model:
    #model = SVC(C=n, kernel = 'linear')
    model = SVC(C=n, kernel = 'poly')
    #model = SVC(C=n, kernel = 'rbf')    
    
    model.fit(X_train, y_train)
    predict = model.predict(X_test)
    acc_score = accuracy_score(y_test, predict)
    accuracy_list.append((n,acc_score))    
# create a DF with the list
acc_score_df = pd.DataFrame(accuracy_list, columns=['C', 'Score'])
# print the DF by score
print(acc_score_df.sort_values('Score', ascending=False))

# tfidf vectorized dataframe - crossvalidation
model1 = MultinomialNB()
model2 = SVC(C=4, kernel = 'linear')
model3 = SVC(C=1, kernel = 'poly')
model4 = SVC(C=3, kernel = 'rbf')

print("tfidf - MNB:", np.average(cross_val_score(model1, X_train, y_train, cv=5)))
print("tfidf - SVC - linear:", np.average(cross_val_score(model2, X_train, y_train, cv=5)))
print("tfidf - SVC - poly:", np.average(cross_val_score(model3, X_train, y_train, cv=5)))
print("tfidf - SVC - rbf:", np.average(cross_val_score(model4, X_train, y_train, cv=5)))

#####################################################################################################

#function used to create classifiers from data frame, labels, and the vectorizer used
def model_prediction(X, vectorizer):


    #create test and training using 60% for training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=2)
    
    unique, counts = np.unique(y_train, return_counts=True)
    print("train\n", np.asarray((unique, counts)))
    unique, counts = np.unique(y_test, return_counts=True)
    print("\ntest\n", np.asarray((unique,counts)))
    #classification reports and matrices
    target_names = ['0','1', '2']
    
    
    MyModelNB= MultinomialNB()
    NB1=MyModelNB.fit(X_train, y_train)
    prediction = MyModelNB.predict(X_test)
    print("MNB")
    print(classification_report(y_test, prediction, target_names=target_names))
    
    print("accuracy:", NB1.score(X_test,y_test), "\n")
    
    score = '{:.2%}'.format(NB1.score(X_test,y_test))
    print("accuracy", score)

    # confusion matrix
    cv_cm = confusion_matrix(y_test, prediction)
    print(cv_cm)
    # plot the confusion matrix using a heatmap
    df_cv_cm = pd.DataFrame(cv_cm, index=NB1.classes_ , columns=NB1.classes_)
    sns.heatmap(df_cv_cm, cmap='Blues', annot=True, fmt='g')
    plt.title('MNB - Confusion Matrix - Frequency\nAccuracy:'+str(score))
    plt.show()
    
                            
    #svm models
    #need to update "C" for tf and tfidf vectorized dataframes:
    svm_linear = SVC(C=4, kernel = 'linear')
    svm_linear_pred = svm_linear.fit(X_train, y_train).predict(X_test)
    
    svm_poly = SVC(C=1, kernel = 'poly')
    svm_poly_pred = svm_poly.fit(X_train, y_train).predict(X_test)

    svm_rbf = SVC(C=3, kernel = 'rbf')
    svm_rbf_pred = svm_rbf.fit(X_train, y_train).predict(X_test)

    print("SVM linear")
    print(classification_report(y_test, svm_linear_pred, target_names=target_names))

    print("\nSVM linear Confusion Matrix\n")
    print(confusion_matrix(y_test, svm_linear_pred, labels=[0,1,2]), "\n\n")
    
    print("accuracy:", svm_linear.score(X_test,y_test), "\n")
    
    score = '{:.2%}'.format(svm_linear.score(X_test,y_test))
    print("accuracy", score)
    
    # confusion matrix for linear kernel
    cv_cm = confusion_matrix(y_test, svm_linear_pred)
    print(cv_cm)
    # plot the confusion matrix using a heatmap
    df_cv_cm = pd.DataFrame(cv_cm, index=svm_linear.classes_ , columns=svm_linear.classes_)
    sns.heatmap(df_cv_cm, cmap='Purples', annot=True, fmt='g')
    plt.title('SVM linear - Confusion Matrix - Frequency\nAccuracy:'+str(score))
    plt.show()

    print("SVM poly")
    print(classification_report(y_test, svm_poly_pred, target_names=target_names))

    print("\nSVM pol Confusion Matrix\n")
    print(confusion_matrix(y_test, svm_poly_pred, labels=[0,1,2]), "\n\n")
    
    print("accuracy:", svm_poly.score(X_test,y_test), "\n")
    
    score = '{:.2%}'.format(svm_poly.score(X_test,y_test))
    print("accuracy", score)
    
    # confusion matrix for poly kernel
    cv_cm = confusion_matrix(y_test, svm_poly_pred)
    print(cv_cm)
    # plot the confusion matrix using a heatmap
    df_cv_cm = pd.DataFrame(cv_cm, index=svm_poly.classes_ , columns=svm_poly.classes_)
    sns.heatmap(df_cv_cm, cmap='Purples', annot=True, fmt='g')
    plt.title('SVM poly - Confusion Matrix - Frequency\nAccuracy:'+str(score))
    plt.show()

    print("SVM rbf")
    print(classification_report(y_test, svm_rbf_pred, target_names=target_names))

    print("\nSVM rbf Confusion Matrix\n")
    print(confusion_matrix(y_test, svm_rbf_pred, labels=[0,1,2]), "\n\n")
    
    print("accuracy:", svm_rbf.score(X_test,y_test), "\n")
    
    score = '{:.2%}'.format(svm_rbf.score(X_test,y_test))
    print("accuracy", score)
    
    # confusion matrix for rbf kernel
    cv_cm = confusion_matrix(y_test, svm_rbf_pred)
    print(cv_cm)
    # plot the confusion matrix using a heatmap
    df_cv_cm = pd.DataFrame(cv_cm, index=svm_rbf.classes_ , columns=svm_rbf.classes_)
    sns.heatmap(df_cv_cm, cmap='Purples', annot=True, fmt='g')
    plt.title("SVM rbf - Confusion Matrix - Frequency\nAccuracy:"+str(score))
    plt.show()
    
    ##MNB model
    print("MNB model indicative words\n\n")
    
    #indicative negative words
    print("indicative negative words\n")
    show_most_and_least_informative_features(vectorizer, MyModelNB, class_idx=0, n=10)
    
    #indicative neutral words
    print("\n\nindicative neutral words\n")
    show_most_and_least_informative_features(vectorizer, MyModelNB, class_idx=1, n=10)
    
    #indicative positive words
    print("\n\nindicative positive words\n")
    show_most_and_least_informative_features(vectorizer, MyModelNB, class_idx=2, n=10)
    
    
    #svm linear model
    print("\n\nSVM linear model indicative words\n\n")
    
    #indicative negative words
    print("indicative negative words\n")
    show_most_and_least_informative_features(vectorizer, svm_linear, class_idx=0, n=10)
    
    #indicative neutral words
    print("\n\nindicative neutral words\n")
    show_most_and_least_informative_features(vectorizer, svm_linear, class_idx=1, n=10)
    
    #indicative positive words
    print("\n\nindicative positive words\n")
    show_most_and_least_informative_features(vectorizer, svm_linear, class_idx=2, n=10)
    


#call function
#model_prediction(MyDTM_DF_CV, MyCV)
model_prediction(MyDTM_DF_Tfidf, MyTfidfV)


###############################################################################################
#LDA for Topic Modeling Section

from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text 
from sklearn.feature_extraction.text import TfidfVectorizer

#stop_word = text.ENGLISH_STOP_WORDS.union(["en", "climatechangeisreal", "actonclimate", "feelthebern", "mashablenews","rt", "dicaprio", "cc","oscars", "tcot","th", "hour", "climateemergency", "climateaction","dont", "read", "just", "time", "climate", "global", "warming", "https", "climatechange", "amp", "make", "ll", "adb", "leonardodicaprio", "leodicaprio", "https", "globalwarming", "change", "action", "changes", "li", "sn", "musk", "elon", "lo", "ol", "faa"])
stop_word = text.ENGLISH_STOP_WORDS.union(["saveourwaves", "hng", "urbanga", "therevenant", "carbonfootprint", "mashable", "greenpeace", "elchatoprada", "susansarandon", "lanaparrilla", "jack", "iamsuperbianca", "juanamartinezh", "el", "tcot", "uniteblue", "actonclimate", "cc", "dulcemaria", "feelthebern", "gogreen", "youtube", "pm", "mt", "keepitintheground", "fossilfuel", "auspol", "th", "congratsleo", "marcorubio", "berniesanders", "qu", "twitter", "adaption", "ab", "addressing", "affects", "affected", "hate", "abt", "fuck", "rt",  "mashablenews", "en", "di", "si", "need", "needed", "climateaction", "climatechangeisreal", "oscars","leo", "leonardo", "dicaprio", "https", "climatechange", "amp", "make", "ll", "adb", "leonardodicaprio", "leodicaprio", "https", "globalwarming", "change", "action", "changes", "li", "sn", "musk", "elon", "lo", "ol", "faa"])
          
#vectorizer
MyCountV = CountVectorizer(input="content", min_df=2, lowercase=True, max_features = 1000, stop_words = stop_word)
MyTfidfV = TfidfVectorizer(input="content", ngram_range=(2, 2), use_idf=True, min_df=2,  stop_words= stop_word , max_features = 1000, lowercase=True)

#function to use LDA for topic modeling of each list
def lda_function(newList, vectorizer):

    #sparse matrix
    MyDTM = vectorizer.fit_transform(newList)

    #list of vocabulary
    vocab = vectorizer.get_feature_names()

    #create df
    MyDTM_DF = pd.DataFrame(MyDTM.toarray(), columns = vocab)
    #print(MyDTM_DF)
    
    #LDA for topic modeling
    num_topics = 3

    lda_model_DH = LatentDirichletAllocation(n_components=num_topics, 
                                             max_iter=1000, learning_method='online')

    LDA_DH_Model = lda_model_DH.fit_transform(MyDTM_DF)

    #create function for top words in decreasing order 
    def print_topics(model, vectorizer, top_n=10):
        for idx, topic in enumerate(model.components_):
            print("Topic:  ", idx)
            print([(vectorizer.get_feature_names()[i], topic[i])
                            for i in topic.argsort()[:-top_n - 1:-1]])

    #call the function
    print_topics(lda_model_DH, vectorizer, 15)

    #visualization of each topic
    import matplotlib.pyplot as plt
    import numpy as np
    word_topic = np.array(lda_model_DH.components_)

    #print(type(word_topic))

    word_topic = word_topic.transpose()

    num_top_words = 10 ##
    vocab_array = np.asarray(vocab)
    #fontsize_base = 70 / np.max(word_topic) 
    fontsize_base = 10

    for t in range(num_topics):
        plt.subplot(1, num_topics, t + 1)  # plot numbering starts with 1
        plt.ylim(0, num_top_words + 0.5)  # stretch the y-axis to accommodate the words
        plt.xlim(0, num_top_words + 1)  # stretch the y-axis to accommodate the words
        plt.xticks([])  # remove x-axis markings ('ticks')
        plt.yticks([]) # remove y-axis markings ('ticks')
        plt.title('Topic #{}'.format(t))
        top_words_idx = np.argsort(word_topic[:,t])[::-1]  # descending order
        top_words_idx = top_words_idx[:num_top_words]
        top_words = vocab_array[top_words_idx]
        top_words_shares = word_topic[top_words_idx, t]
        for i, (word, share) in enumerate(zip(top_words, top_words_shares)):
            plt.text(0.3, num_top_words-i-0.5, word, fontsize=fontsize_base)

    plt.tight_layout()
    plt.show()

    
#call the LDA function:

#lda_function(clean_text[0:1000], MyCountV)
#lda_function(clean_text, MyCountV)
lda_function(NeutralList, MyTfidfV)


###############################################################################################################
#API to gather tweets
#!/usr/bin/env python
# coding: utf-8

# In[1]:


### 

##TwitterMining_Token_WordCloud.py

###Packages-----------------------



import tweepy

#conda install -c conda-forge tweepy

from tweepy import OAuthHandler

import json

from tweepy import Stream

from tweepy.streaming import StreamListener

import sys



import json

from nltk.tokenize import word_tokenize

from nltk.tokenize import TweetTokenizer

import re



from os import path

#from scipy.misc import imread

import matplotlib.pyplot as plt

##install wordcloud

## conda install -c conda-forge wordcloud

## May also have to run conda update --all on cmd

#import PIL

#import Pillow

#import wordcloud

from wordcloud import WordCloud, STOPWORDS

###-----------------------------------------


# In[5]:


consumer_key = "knVP60l36N2sMZHhV10do3dC3"
consumer_secret = "Mg8KwH7h30bimA5J0qfkqjab9OqwKHRkUD3cQRDwGwScJDnan5"
access_token = "2742745332-hPW39w5MRLTAzOcXPYVKzpf7nuxxIhRC4xn9u0s"
access_secret = "a13W4EsMvb3wPLDwTaHpY6noLdR4rsmd1fKtRSHw1pIml"


# In[3]:


auth = OAuthHandler(consumer_key, consumer_secret)


# In[6]:


auth.set_access_token(access_token, access_secret)


# In[7]:


api = tweepy.API(auth)


# In[ ]:





# In[8]:


class Listener(StreamListener):

    print("In Listener...") 

    tweet_number=0

    #__init__ runs as soon as an instance of the class is created

    def __init__(self, max_tweets, hfilename, rawfile):

        self.max_tweets=max_tweets

        print(self.max_tweets)     

    #on_data() is a function of StreamListener as is on_error and on_status    

    def on_data(self, data):

        self.tweet_number+=1 

        print("In on_data", self.tweet_number)

        try:

            print("In on_data in try")

            with open(hfilename, 'a') as f:

                with open(rawfile, 'a') as g:

                    tweet=json.loads(data)

                    tweet_text=tweet["text"]

                    print(tweet_text,"\n")

                    f.write(tweet_text) # the text from the tweet

                    json.dump(tweet, g)  #write the raw tweet

        except BaseException:

            print("NOPE")

            pass

        if self.tweet_number>=self.max_tweets:

            #sys.exit('Limit of '+str(self.max_tweets)+' tweets reached.')

            print("Got ", str(self.max_tweets), "tweets.")

            return False

    #method for on_error()

    def on_error(self, status):

        print("ERROR")

        if(status==420):

            print("Error ", status, "rate limited")

            return False

#----------------end of class Listener


# In[12]:


hashname=input("Enter a hashtag to search:  ") 

numtweets=eval(input("How many tweets do you want to get?: "))

print(numtweets)

if(hashname[0]=="#"):

    nohashname=hashname[1:] #remove the hash

else:

    nohashname=hashname

    hashname="#"+hashname

#print(hashname)

#Create a file for any hash name    

hfilename="file_"+nohashname+".txt"

print(hfilename)

rawfile="file_rawtweets_"+nohashname+".txt"

print(rawfile)

twitter_stream = Stream(auth, Listener(numtweets, hfilename, rawfile))

#twitter_stream.filter(track=['#womensrights'])

twitter_stream.filter(track=[hashname])

print("Twitter files created....")


# In[ ]:


#climatechange, #globalwarming, #parisagreement:










