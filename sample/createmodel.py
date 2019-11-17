#importing all the necessary libraries
import pystempel
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import warnings
from stempel import StempelStemmer
warnings.filterwarnings("ignore", category=DeprecationWarning)

%matplotlib inline

#loading the dataframes
train_df = pd.read_csv('polit_tweets.csv')
test_df = pd.read_csv('tweets_no_label.csv')

#combining dataframes into one general dataframe
combined_df = train_df.append(test_df,ignore_index=True,sort=True)

combined_df.info()

#declaring a function to find and remove certain patterns using regex
def remove_pattern(text,pattern):

    #re.findall() finds the patterns, eg. @user, and puts it in a list for further task
    r = re.findall(pattern,text)

    #re.sub() removes pattern from the sentences in the dataset
    for i in r:
        text = re.sub(i,"",text)

    return text

#removing '@user' pattern through vectorize function
combined_df['Tidy_Tweets'] = np.vectorize(remove_pattern)(combined_df['Text'], "@[\w]*")

#removing 'https://t.co/' pattern through vectorize function
#since in our database there are tweets with links, we need to remove links as well
combined_df['Tidy_Tweets'] = np.vectorize(remove_pattern)(combined_df['Tidy_Tweets'], "https://t.co/[\w]*")

#replacing the symbols and punctuation with spaces, leaving all the polish special characters in place
combined_df['Tidy_Tweets'] = combined_df['Tidy_Tweets'].str.replace("[^\w]", " ")

#getting rid of all the numbers in the tweets
combined_df['Tidy_Tweets'] = combined_df['Tidy_Tweets'].str.replace("[0-9]", " ")

#removing '#hashtag' pattern, since Polish politicians rarely, if ever, use hashtags and it makes more sense to get rid of them
combined_df['Tidy_Tweets'] = np.vectorize(remove_pattern)(combined_df['Tidy_Tweets'], "#[\w]*")

#tokenizing the tweets
tokenized_df = combined_df['Tidy_Tweets'].apply(lambda x: x.split())

#reading a .txt list of stopwords into a python list (stopwords were taken from https://github.com/bieli/stopwords)
with open('polishstopwords.txt', 'r') as stopwords:
    stop = stopwords.read().splitlines()

#getting rid of the stop words
for i in range(len(tokenized_df)):
    tokenized_df[i] = [j for j in tokenized_df[i] if j not in stop]

#stemming the tweets (stripping the suffixes) using pystempel library
ps = StempelStemmer.polimorf()
tokenized_df = tokenized_df.apply(lambda x: [ps.stem(i) for i in x])

#Polish stemmer is defective and sometimes turns random words into NoneType, which prevents stiching the tokens back into strings
#we need to remove all NoneType objects in order to proceed
for i in range(len(tokenized_df)):
     tokenized_df[i] = [j for j in tokenized_df[i] if j]

#stiching the tweets back together
for i in range(len(tokenized_df)):
    tokenized_df[i] = ' '.join(tokenized_df[i])

combined_df['Tidy_Tweets'] = tokenized_df

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
wc.to_file('N.png')

#to extract features from the tweets, we're going to use Bag of Words method
from sklearn.feature_extraction.text import CountVectorizer

bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')

bow = bow_vectorizer.fit_transform(combined_df['Tidy_Tweets'])

bow_df = pd.DataFrame(bow.todense())

bow_df

bow_train = bow[:1249]

bow_train.todense()

# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(bow_train,train_df['label'],test_size=0.3,random_state=109)
X_train=X_train.toarray()
X_test=X_test.toarray()

#since there isn't too much data to process and the whole thing is rather simple, I've decided to use Naive Bayes Algorithm for classification
#our source of knowledge about Native Bayes Algorithm was https://www.datacamp.com/community/tutorials/naive-bayes-scikit-learn

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
