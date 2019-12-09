#importing all the necessary libraries
import re
import pandas as pd
import numpy as np
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

#to extract features from the tweets, we're going to use TF-IDF method
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_df=0.90, min_df=2,max_features=1000)

tfidf_matrix = tfidf.fit_transform(combined_df['Tidy_Tweets'])

df_tfidf = pd.DataFrame(tfidf_matrix.todense())

train_tfidf_matrix = tfidf_matrix[:1249]

# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_valid, y_train, y_valid = train_test_split(train_tfidf_matrix,train_df['label'],test_size=0.3,random_state=17)

#we use Logistic Regression for classification, because it provides fairly good accuracy
#Import Logistic Regression model
from sklearn.linear_model import LogisticRegression
Log_Reg = LogisticRegression(random_state=0,solver='lbfgs')

Log_Reg.fit(X_train,y_train)

prediction_tfidf = Log_Reg.predict_proba(X_valid)

prediction_int = prediction_tfidf[:,1]>=0.4

prediction_int = prediction_int.astype(np.int)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(y_valid, prediction_int))

test_tfidf = tfidf_matrix[:19572]
test_pred = Log_Reg.predict_proba(test_tfidf)

test_pred_int = test_pred[:,1] >= 0.4
test_pred_int = test_pred_int.astype(np.int)

results_df = pd.DataFrame({'label': test_pred_int})

submission = results_df[['label']]
submission['tweet'] = test_df['Text']
#some of the cells in a 'tweet' column are empty, because tweets in a train dataframe were also in a test dataframe, and now the duplicats are gone but labels for them are not
#the reason why we didn't get rid of the duplicates earlier is that the function in pandas for erasing duplicates is defective and for some reason deletes more rows than it should
submission.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
submission.info()
submission.to_csv('result.csv', index=False, header=True)

pickle.dump(Log_Reg, open('model.sav', 'wb'))
