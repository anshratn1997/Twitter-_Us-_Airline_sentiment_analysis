
#*** 
 Twitter airline sentiment analysis..
 @author
 Ratnesh keshari
 ***

import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from nltk import pos_tag
from nltk.corpus import wordnet
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk import NaiveBayesClassifier




data=pd.read_csv('C:/Users/ansh41/Desktop/mypython/codingNinja/datasets/training_twitter_x_y_train.csv')

test_data=pd.read_csv('C:/Users/ansh41/Desktop/mypython/codingNinja/datasets/test_twitter_x_test.csv')




y_train=data['airline_sentiment']
x_train=data['text']

x_test=test_data['text']

airline=['@SouthwestAir', '@united', '@JetBlue', '@USAirways', '@AmericanAir',
       '@VirginAmerica','']
tag=set(airline)

stops=set(stopwords.words('english'))
punct=list(string.punctuation)
stops.update(punct)
stops.update(tag)
len(stops)

all_tweets=[''.join(tweet) for tweet in x_train]

cv=CountVectorizer(max_features=200,ngram_range=(1,3),stop_words=stops)

tfcv = TfidfVectorizer(min_df=0.1, max_df = 0.9, sublinear_tf=True, use_idf =True, stop_words =stops,max_features=2000)

x_train_features=tfcv.fit_transform(x_train)

x_train_features.todense()

tfcv.get_feature_names()

x_test_features=tfcv.transform(x_test)

rfc=RandomForestClassifier()
rfc.fit(x_train_features,y_train)
y_pred=rfc.predict(x_test_features)

DTC=DecisionTreeClassifier()
DTC.fit(x_train_features,y_train)
y_pred=DTC.predict(x_test_features)


knc=KNeighborsClassifier()
knc.fit(x_train_features,y_train)
y_pred=knc.predict(x_test_features)





gnb=GaussianNB()
gnb.fit(x_train_features.toarray(),y_train)
y_pred=gnb.predict(x_test_features.toarray())





np.savetxt('ff.csv',y_pred,fmt='%s')

#convert data into nltk format to apply nltk Naivebayes Classifier

documents=[ (tweet.split(" "),sentiment) for tweet,sentiment in zip(x_train,y_train)]
test_documents=[tweet.split(" ") for tweet in x_test]


# To return postag to lemmatizer

def get_simple_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN





lemmatizer=WordNetLemmatizer()



def clean_review(words):
    output_words=[]
    for word in words:
        if word.lower() not in stops:    
            pos=pos_tag([word])
            clean_word=lemmatizer.lemmatize(word,get_simple_pos(pos[0][1]))
            output_words.append(clean_word)
    return output_words




documents=[(clean_review(document),category) for document,category in documents]
test_documents=[clean_review(document) for document in test_documents]
len(test_documents)


all_words=[]
for document in documents:
    all_words+=document[0]


dictionary= nltk.FreqDist(all_words)


common=dictionary.most_common(4000)
features=[i[0] for i in common]

def get_feature_dict(words):
    current_features={}
    word_set=set(words)
    for w in features:
        current_features[w]=w in word_set
    return current_features

training_data=[(get_feature_dict(doc),category) for doc,category in documents]
testing_data=[get_feature_dict(doc) for doc in test_documents]

clf=NaiveBayesClassifier.train(training_data)
clf.show_most_informative_features(50)

y_pred=[]
for i in range(len(testing_data)):
    y_pred.append(clf.classify(testing_data[i]))

np.savetxt('ff.csv',y_pred,fmt='%s')

