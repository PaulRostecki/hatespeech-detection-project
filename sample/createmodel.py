#importing all the necessary libraries
import re
import pandas as pd
import numpy as np
import seaborn as sns
import string
import warnings
from stempel import StempelStemmer
import pickle
from cleaner import cleanall
warnings.filterwarnings("ignore", category=DeprecationWarning)

#loading the dataframes
#as I'm not sharing my own dataset, you'll need to insert your dataset here for the whole thing to work
train_df = pd.read_csv('../lib/train_example.csv')
test_df = pd.read_csv('../lib/test_example.csv')

#combining dataframes into one general dataframe
combined_df = train_df.append(test_df,ignore_index=True,sort=True)

#using a function imported from cleaner.py to clean, tokenize and stem the tweets
combined_df['Tidy_Tweets'] = cleanall(combined_df['Text'])

#importing wordcloud to spit out the wordcloud for the negative words in order to analyze
import wordcloud
from wordcloud import WordCloud,ImageColorGenerator
from PIL import Image
import urllib
import requests

#storing all the negative words from the dataset
#we're doing a wordcloud chart only for negative tweets, there's no point to make any other chart if we're dividing tweets only into negative and not negative
negative = ' '.join(text for text in combined_df['Tidy_Tweets'][combined_df['label']==1])

#combining the image with the dataset
Mask = np.array(Image.open(requests.get('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png', stream=True).raw))

#taking the color of the image and imposing it it over our wordcloud using ImageColorGenerator
image_colors = ImageColorGenerator(Mask)

#saving the wordcloud
wc = WordCloud(background_color='white', height=1500, width=4000,mask=Mask).generate(negative)
wc.to_file('wordcloud.png')

#to extract features from the tweets, we're going to use Bag of Words method
from sklearn.feature_extraction.text import CountVectorizer

bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')

bow = bow_vectorizer.fit_transform(combined_df['Tidy_Tweets'])

bow_df = pd.DataFrame(bow.todense())

bow_train = bow[:1249]

bow_train.todense()

# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(bow_train,train_df['label'],test_size=0.3,random_state=109)
X_train=X_train.toarray()
X_test=X_test.toarray()

#since there isn't too much data to process and the whole thing is rather simple, we use Naive Bayes Algorithm for classification
#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = gnb.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

test_final = bow[:19573]
test_final = test_final.toarray()

test_pred = gnb.predict(test_final)

results_df = pd.DataFrame({'label': test_pred})

submission = results_df[['label']]
submission['tweet'] = test_df['Text']
#some of the cells in a 'tweet' column appear to be empty for some weird reason, so we need to erase the rows which contain empty cells
submission.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
submission.to_csv('result.csv', index=False, header=True)

filename = 'model.sav'
pickle.dump(gnb, open(filename, 'wb'))
