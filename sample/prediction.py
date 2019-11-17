#this code is not really working yet, just working on it
import pickle
import re
import pandas as pd
import numpy as np
import seaborn as sns
import string
import warnings
from cleaner import cleanall
from stempel import StempelStemmer

loaded_model = pickle.load(open('model.sav', 'rb'))
sample = input("Enter the sentence: ")

sample = cleanall(sample)

from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=1.0, min_df=1.0, max_features=1000)
bow = bow_vectorizer.fit_transform(sample)
bow_df = pd.DataFrame(bow.todense())

result = int(loaded_model.predict(bow_df))

if result == 1:
    print('This text may potentially be hatespeech.')
else:
    print('This text is most probably not hatespeech.')
