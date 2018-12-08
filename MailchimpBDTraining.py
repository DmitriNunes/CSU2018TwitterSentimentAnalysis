import pandas as pd
import numpy as np
import csv
import nltk
import string
import matplotlib.pyplot as plt
# matplotlib inline
import seaborn as sns
import random
import os

nltk.download('stopwords')

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer

import pickle
import re
from collections import Counter
from string import punctuation


import cufflinks as cf
cf.go_offline()

#def load_tweets(file):

pd.set_option('display.max_colwidth', -1)
plt.style.use('seaborn-white')

# load train data
input_file = os.path.join('Sentiment_Analysis_Dataset.csv')
data = pd.read_csv(input_file, error_bad_lines=False, encoding='latin-1')
data.columns = ['label','id','date','query','user','message']
data.head(2)

data.to_pickle('df_sent140.p') # save dframe to pickle
df_sent140 = pd.read_pickle('df_sent140.p')  # load from pickle

contractions = {
    "ain't": "is not", "amn't": "am not", "aren't": "are not", "can't": "cannot", "could've": "could have", 
    "couldn't": "could not", "daren't": "dare not", "daresn't": "dare not", "dasn't": "dare not", "didn't": "did not", 
    "doesn't": "does not", "don't": "does not", "e'er": "ever", "everyone's": "everyone is", "hadn't": "had not", 
	"hasn't": "has not", "haven't": "have not", "he'd": "he had", "he'll": "he will", "he's": "he has", "he've": "he have",
	"how'd": "how did", "how'll": "how will", "how're": "how are", "how's": "how has", "I'd": "I would", "I'll": "I will",
	"I'm": "I am", "I'm'a": "I am about to", "I'm'o": "I am going to", "I've": "I have", "isn't": "is not", "it'd": "it would",
	"it'll": "it will", "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "may've": "may have",
	"mightn't": "might not", "might've": "might have", "mustn't": "must not", "mustn't've": "must not have", "must've": "must have",
	"needn't": "need not", "ne'er": "never", "o'clock": "of the clock", "o'er": "over", "ol'": "old", "oughtn't": "ought not",
	"shalln't": "shall not", "shan't": "shall not", "she'd": "she had", "she'll": "she will", "she's": "she is", "should've": "should have",
	"shouldn't": "should not", "shouldn't've": "should not have", "somebody's": "somebody has", "someone's": "someone has", "something's": "something is",
	"that'll": "that will", "that're": "that are", "that's": "that is", "that'd": "that would", "there'd": "there would", "there'll": "there will",
	"there're": "there are", "there's": "there is", "these're": "these are", "they'd": "they would", "they'll": "they will", "they're": "they are",
	"they've": "they have", "this's": "this has", "those're": "those are", "'tis": "it is", "'twas": "it was", "wasn't": "was not", "we'd": "we would",
	"we'd've": "we would have", "we'll": "we will", "we're": "we are", "we've": "we have", "weren't": "were not", "what'd": "what did",
	"what'll": "what will", "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is", "where'd": "where did",
	"where're": "where are", "where's": "where is", "where've": "where have", "which's": "which is", "who'd": "who would", "who'd've": "who would have",
	"who'll": "who will", "who're": "who are", "who's": "who is", "who've": "who have", "why'd": "why did", "why're": "why are", "why's": "why does",
	"won't": "will not", "would've": "would have", "wouldn't": "would not", "y'all": "you all", "you'd": "you had", "you'll": "you will",
	"you're": "you are", "you've": "you have"
}

# helper function to clean tweets
def processTweet(tweet):
    # Remove HTML special entities (e.g. &amp;)
    tweet = re.sub(r'\&\w*;', '', tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','',tweet)
    # Remove tickers
    tweet = re.sub(r'\$\w*', '', tweet)
    # To lowercase
    tweet = tweet.lower()
    # Remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*\/\w*', '', tweet)
    # Remove hashtags
    tweet = re.sub(r'#\w*', '', tweet)
    # Replace contractions
    tweet = re.sub(r"(.*'.*) ", lambda m: contractions.get(m.group(), m.group()), tweet)
    # Remove Punctuation and split 's, 't, 've with a space for filter
    tweet = re.sub(r'[' + punctuation.replace('@', '') + ']+', ' ', tweet)
    # Remove words with 2 or fewer letters
    tweet = re.sub(r'\b\w{1,2}\b', '', tweet)
    # Remove whitespace (including new line characters)
    tweet = re.sub(r'\s\s+', ' ', tweet)
    # Remove single space remaining at the front of the tweet.
    tweet = tweet.lstrip(' ') 
    # Remove characters beyond Basic Multilingual Plane (BMP) of Unicode:
    tweet = ''.join(c for c in tweet if c <= '\uFFFF') 
    return tweet
# ______________________________________________________________

# clean dataframe's text column
df_sent140['message'] = df_sent140['message'].apply(processTweet)
# preview some cleaned tweets
df_sent140['message'].head()

# drop duplicates
df_sent140 = df_sent140.drop_duplicates('message')
df_sent140.shape


# tokenize helper function
def text_process(raw_text):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    return [char for char in list(raw_text) if char not in string.punctuation]



def remove_words(word_list):
    remove = ['claytonstateuniversity','clayton state university','clayton state','claytonstate','...','“','”','’','…','mailchimp','mail chimp']
    removed = ' '.join([w for w in word_list if w not in remove])
    return removed

# -------------------------------------------

# tokenize message column and create a column for tokens
df_sent140 = df_sent140.copy()
df_sent140['tokens'] = df_sent140['message'].apply(text_process) # tokenize style 1
df_sent140['message'] = df_sent140['tokens'].apply(remove_words) #tokenize style 2
# df_sent140.head()

# Run Train Data Through Pipeline analyzer=text_process
# uncomment below to train on a larger dataset but it is very slow for a regular laptop

X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2)
#X_train, X_test, y_train, y_test = train_test_split(data['message'][:500], data['label'][:500], test_size=0.2)


# create pipeline
pipeline = Pipeline([
    ('bow', CountVectorizer(strip_accents='ascii',
                            stop_words='english',
                            lowercase=True,
                            )),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

# this is where we define the values for GridSearchCV to iterate over
parameters = {'bow__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'classifier__alpha': (1e-2, 1e-3),
             }

# do 10-fold cross validation for each of the 6 possible combinations of the above params
grid = GridSearchCV(pipeline, cv=10, param_grid=parameters, verbose=1)
grid.fit(X_train,y_train)

# summarize results
print("\nBest Model: %f using %s" % (grid.best_score_, grid.best_params_))
print('\n')
means = grid.cv_results_['mean_test_score']
stds = grid.cv_results_['std_test_score']
params = grid.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("Mean: %f Stdev:(%f) with: %r" % (mean, stdev, param))
	
# save best model to current working directory
joblib.dump(grid.best_estimator_, "twitter_sentiment.pkl")
