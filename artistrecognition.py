
# coding: utf-8

# In[22]:

#import everything we need
import pandas as pd
from random import randint
from random import shuffle
import random
import pickle

import string

import pandas as pd

import nltk
from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import NuSVC

from sklearn.metrics import classification_report as clsr
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.cross_validation import StratifiedShuffleSplit
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
 
import time
import pickle

import numpy as np

import random

import operator


# In[12]:

MIN_SONGS_PER_ARTIST = 30
PATH = '/Users/sherida/Downloads/'


# In[13]:

class NLTKPreprocessor(BaseEstimator, TransformerMixin):
    """
    This class glues NLTK functionality to scikit-learn, so that we can use NLTK in a scikit-learn Pipeline.
    """
    ADDITIONAL_STOP_WORDS = ['verse', 'chorus', 'choru']


    def __init__(self, stopwords=None, punct=None,
                 lower=True, strip=True):
        self.lower      = lower
        self.strip      = strip
        self.stopwords  = stopwords or set(sw.words('english'))
        for add_sw in self.ADDITIONAL_STOP_WORDS:
            self.stopwords.add(add_sw)
        self.punct      = punct or set(string.punctuation)
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        return self

    def inverse_transform(self, X):
        return [" ".join(doc) for doc in X]

    def transform(self, X):
        return [
            list(self.tokenize(doc)) for doc in X
        ]

    def tokenize(self, document):
        char_to_rem = ["\n", "'", ",", "]", "[", ")", "("]
        document = str(document)
        for c in char_to_rem:
            document = document.replace(c, "")

        # Break the document into sentences
        for sent in sent_tokenize(document):
            
            # Break the sentence into part of speech tagged tokens
            for token, tag in pos_tag(wordpunct_tokenize(sent)):
                # Apply preprocessing to the token
                token = token.lower() if self.lower else token
                token = token.strip() if self.strip else token
                token = token.strip('_') if self.strip else token
                token = token.strip('*') if self.strip else token              

                # If stopword, ignore token and continue
                if token in self.stopwords:
                    continue

                # If punctuation, ignore token and continue
                if all(char in self.punct for char in token):
                    continue

                # Lemmatize the token and yield
                lemma = self.lemmatize(token, tag)
                yield lemma

    def lemmatize(self, token, tag):
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)

nltk_pp = NLTKPreprocessor()

def identity(arg):
    """
    Simple identity function works as a passthrough.
    """
    return arg


# In[14]:

#get the raw data
dat = pd.read_csv("/Users/sherida/Downloads/songdata.csv")

# remove artists with very little songs -- these are not interesting
count_by_artist = dat.groupby('artist')["artist"].count()
to_drop = count_by_artist[count_by_artist < MIN_SONGS_PER_ARTIST]

sample = dat[~dat['artist'].isin(to_drop.index)]

#pickles openen
with open(PATH + 'text_final.pickle', 'rb') as file:
    text_classifier=pickle.load(file)

with open(PATH + 'avg_pos_artist.pickle', 'rb') as file:
    avg_pos_artist=pickle.load(file)
    
with open(PATH + 'sentiment.pickle', 'rb') as file:
    sentiment_classifier=pickle.load(file)    


# In[15]:

#get all artists
artists = list(set(sample['artist']))


# In[16]:

def word_feats(words):
    return dict([(word, True) for word in words])

    negids = movie_reviews.fileids('neg')
    posids = movie_reviews.fileids('pos')

    negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
    posfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posids]

    # TODO: check whether cast to int is correct 
    negcutoff = int(len(negfeats)*3/4)
    poscutoff = int(len(posfeats)*3/4)

    trainfeats = negfeats + posfeats
    print('train on %d instances' % (len(trainfeats)))

    sentiment_classifier = NaiveBayesClassifier.train(trainfeats)


# In[17]:

def get_computeranswer(text, options, all_artists):
    # tokenize text
    tokenized_text = nltk_pp.tokenize(text)
    
    # get text model ranks
    #class probs are all probabilities
    class_probs = text_classifier.predict_proba([text])[0]
    class_probs.item(1)

    indeces = []

    class_probs.item(1)
    # filter out the options
        # find the indices of the options in all_artists
    for i in options:
        index = all_artists.index(i)
        indeces.append((i,class_probs.item(index)))
    
    #rankings lopen van achter naar voor... dont know why
    sorted_by_second = sorted(indeces, key=lambda tup: tup[1])
    
    
        
    #rank remaining probabilities
    #predicted_indices = class_probs.argsort()[-4:][::-1]
    
    
    # calculate sentiment
    featureset = (word_feats(tokenized_text))
    result = sentiment_classifier.prob_classify(featureset).prob('pos')
    
    
    
    joe = avg_pos_artist-result
    sorted2 = joe.sort_values()
#     print(sorted2)
#     print(options[1])
#     print(sorted2.get(options[1]))
    
    rank_sentiment = []
    for i,n in enumerate(options):
        waarde = sorted2.get(options[i])
        rank_sentiment.append((n,waarde))
        
    rank_sentiment = sorted(indeces, key=lambda tup: tup[1])
    
    final_dict={}
    for i,n in enumerate(sorted_by_second):
#         print(" HALLO")
#         print(i,n)
        final_dict[n[0]] = i
    
    for i,n in enumerate(rank_sentiment):
#         print("WACHT")
#         print(i,n)
        final_dict[n[0]] += i
    
    antwoord = max(final_dict, key=lambda key: final_dict[key])
        
    # get sentiment ranks
        # calculate differences between sentiment and option sentiment avgs
        # rank differences
    
    # sum ranks
    
    # select lowest number, or random out of lowest numbers
    return (antwoord)


# In[ ]:




# In[18]:

def getartist(trueartist, artists, numberofartists):
    artistlist = list(artists)
    artistlist.remove(trueartist)
    chosenartists = [trueartist]
    for i in range(0,numberofartists):
        addartist = artistlist[randint(0,len(artistlist)-1)]
        chosenartists.append(addartist)
        artistlist.remove(addartist)
    return chosenartists


# In[19]:

#start the game

print("Can you beat the computer?")
print("A game about songs and artists")


# In[21]:

#handle the rounds
gamelength = 10000 #rounds
computerscore = 0;
playerscore = 0;
for i in range(0, gamelength):
    randomnumber = randint(0, len(sample)-1)
    artist = sample['artist'].get(randomnumber, "DEFAULT")
    if (artist == "DEFAULT"):
        print("default:", gamelength +1)
        gamelength += 1
        continue
    text = sample['text'][randomnumber]
    roundartists = getartist(artist, artists, 3)
    shuffle(roundartists)
    computeranswer = get_computeranswer(text, roundartists, artists)

    
    if (roundartists[0] == artist):
        correctanswer = "A"
    elif (roundartists[1] == artist):
        correctanswer = "B"
    elif (roundartists[2] == artist):
        correctanswer = "C"
    else:
        correctanswer = "D"
        
    if (roundartists[0] == computeranswer):
        computeranswer = "A"
    elif (roundartists[1] == computeranswer):
        computeranswer = "B"
    elif (roundartists[2] == computeranswer):
        computeranswer = "C"
    else:
        computeranswer = "D"
    

    
    if (computeranswer == correctanswer):
        computerscore += 1
    print("game:", i, "score:", computerscore)

print("total:", computerscore)
    



