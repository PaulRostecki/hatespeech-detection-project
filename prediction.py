#importing all the necessary libraries
import pickle
import pandas as pd
import numpy as np
import string
from cleaner import cleanall

#importing the previously built model
loaded_model = pickle.load(open('model.sav', 'rb'))

#merging the given string with the training df, for TF-IDF to work and correctly interpret the data
#as I'm not sharing my own dataset, you'll need to insert your dataset here for the whole thing to work
sample = [input("Enter the sentence: ")]
train_df = pd.read_csv('../lib/train_example.csv')
input_df = pd.DataFrame(sample, columns = ['Text'])
input_df['label'] = np.nan
combined_df = train_df.append(input_df,ignore_index=True,sort=False)

#using a function imported from cleaner.py to clean, tokenize and stem the tweets
cleanall(combined_df['Text'])

#using TF-IDF method to extract features from the tweets
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=1000)
df_matrix = tfidf.fit_transform(combined_df['Text'])
df_matrix = pd.DataFrame(df_matrix.todense())

#predicting the probability of the text being a hatespeech
result = loaded_model.predict_proba(df_matrix)

samplenumber = len(result)-1

#if the model is at least 40% sure that may be a hatespeech, it labels it as hatespeech
result = result[:,1] >= 0.4

if result[samplenumber] == 1:
    print('This text may potentially be hatespeech.')
else:
    print('This text is most probably not hatespeech.')
