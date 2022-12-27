import re
import numpy as np
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import nltk

nltk.download('stopwords')

def parseANDprocess_feedback(feedback):
    """Parses the given comment and outputs a simplified list containing words as in the original feedback"""
    
    
    #use porter stemmer algorithm for stemming
    stemmer=PorterStemmer()
    stopwords_english=stopwords.words('english')
    #using regular expressions to remove hyperlinks and hashtags
    feedback=re.sub(r'https?:\/\/.*[\r\n]*', '', feedback)
    feedback=re.sub(r'#', '', feedback)
    #decompose the tweet into a list of tokens using nltk tokenizer method
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,reduce_len=True)
    feedback_tokens=tokenizer.tokenize(feedback)
    
    #parsed feedback is the list of tokens after simplification (removing stopwords)
    parsed_feedback = []
    for word in feedback_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            parsed_feedback.append(stem_word)

    return parsed_feedback

def build_freqs(feedbacks, ys):
    """Build frequencies.
    Input:
        feedbacks: a list of feedbacks
        ys: an m x 1 array with the sentiment label of each tweet
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its
        frequency
    """
    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element.
    # Also note that this is just a NOP if ys is already a list.
    yslist = np.squeeze(ys).tolist()

    # Start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet.
    freqs = {}
    for y, feedback in zip(yslist, feedbacks):
        for word in parseANDprocess_feedback(feedback):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs

def feature_extraction(feedback,freqs):
   
   word_list=parseANDprocess_feedback(feedback)
   x=np.zeros((1,3))
   x[0,0]=1
   
   for each_word in word_list :
      x[0,1]+=freqs.get((each_word,'positive'),0)
      x[0,2]+=freqs.get((each_word,'negative'),0)
   
   assert(x.shape == (1, 3))
   return x
