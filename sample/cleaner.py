import re
import pandas as pd
import numpy as np
import string
from stempel import StempelStemmer

def cleanall(df):

    #converting given variable to Series for further cleaning
    df = pd.Series(df)

    #making all letters lowercase, to avoid further issues regarding case sensivity
    df = df.str.lower()

    #declaring a function to find and remove certain patterns using regex
    def remove_pattern(text,pattern):

        #re.findall() finds the patterns, e.g. @user, and puts it in a list for further task
        r = re.findall(pattern,text)

        #re.sub() removes pattern from the sentences in the dataset
        for i in r:
            text = re.sub(i,"",text)

        return text

    #removing '@user' pattern through vectorize function
    df = np.vectorize(remove_pattern)(df, "@[\w]*")

    #removing 'https://t.co/' pattern through vectorize function
    #since in our database there are tweets with links, we need to remove links as well
    df = np.vectorize(remove_pattern)(df, "https://t.co/[\w]*")

    df = pd.Series(df)

    #replacing the symbols and punctuation with spaces, leaving all the polish special characters in place
    df = df.str.replace("[^\w]", " ")

    #getting rid of all the numbers in the tweets
    df = df.str.replace("[0-9]", " ")

    #removing '#hashtag' pattern, since in our database hashtags are rarely used and it makes more sense to just get rid of them
    df = np.vectorize(remove_pattern)(df, "#[\w]*")

    df = pd.Series(df)

    #tokenizing the tweets
    df = df.apply(lambda x: x.split())

    #reading a .txt list of stopwords into a python list (stopwords were taken from https://github.com/bieli/stopwords)
    with open('../lib/polishstopwords.txt', 'r') as stopwords:
        stop = stopwords.read().splitlines()

    #getting rid of the stop words
    for i in range(len(df)):
        df[i] = [j for j in df[i] if j not in stop]

    #stemming the tweets (stripping the suffixes) using pystempel library
    ps = StempelStemmer.polimorf()
    df = df.apply(lambda x: [ps.stem(i) for i in x])

    #Polish stemmer is defective and sometimes turns random words into NoneType, which prevents stiching the tokens back into strings
    #we need to remove all NoneType objects in order to proceed
    for i in range(len(df)):
         df[i] = [j for j in df[i] if j]

    #stiching the tweets back together
    for i in range(len(df)):
        df[i] = ' '.join(df[i])

    return df
