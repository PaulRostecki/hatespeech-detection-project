# Machine Learning Hatespeech Detection - Intel AI for Youth project
### Intro
Hatespeech Detection project is aimed to analyze and detect hatespeech written in Polish language. The repository contains my trained model, as well as the script which generates a model and the script that predicts if the text is hatespeech or not, based on the given model. Unfortunately, it does not contain any datasets, so you need to insert your own data to make everything work. All the current scripts that require the .csv datasets, refer to the example files.

It's also worth mentioning, that the code and the model contained in that repository are actually the basis for the [MakeLoveNotHate](http://www.makelovenothate.ml/) webapp, which was submitted to the final part of the Intel AI4Y Project.

## The repository consists of:

#### 1. folder *lib*
A folder that contains all the databases (`train_example.csv`, `test_example.csv`) and `polishstopwords.txt`, which contains stopwords that are later erased from the entries in the dataset.

As the topic of hatespeech is very fragile and controversial, I'm not sharing my own datasets used in the process. The .csv files in the folder are empty, and show only the general form of the database that you should follow when building your own base. All the texts in the train dataset are labeled as either 0 (neutral/positive text) or 1 (negative text/hatespeech), while in test dataset, the column `label` remains empty.

#### 2. createmodel.py
A script that creates a model based on the data from the `.csv` files. The text is converted and weighted using the [TF-IDF](https://en.wikipedia.org/wiki/Tf–idf) method, and then the model is trained based on the [Logistic Regression](https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html) classification algorithm. The script also makes a wordcloud with the most used negative words and predicts the labels for the whole test dataset. It also saves the labeled dataset, the wordcloud in `.png` and a trained model.

#### 3. cleaner.py
A script, imported by the other scripts, that cleans the text in the dataset. It gets rid of the special characters, numbers, etc., [tokenizes](https://en.wikipedia.org/wiki/Lexical_analysis#Tokenization), then [stems](https://en.wikipedia.org/wiki/Stemming) all the texts and at the end, returns the cleaned texts to the dataset for further conversion.
##### Important note!
PyStempel, a stemmer used in the cleaner.py is made to work with Polish language and won't work with other languages. If you wish to analyze texts from another languages, you need to import another stemmer.

#### 4. prediction.py 
A script that takes input from a user in a form of a sentence, then attaches it into the dataset, cleans it, and gives a prediction based on a loaded model, trained earlier in `createmodel.py`. Note that you need to use the whole dataset in the process, as the TF-IDF method applies weights basing on the frequency in the whole document (or dataset in this case).

## Sources:
The list of stopwords was taken from [this repository](https://github.com/bieli/stopwords)

The whole project was initially based on the tutorial from [towardsdatascience.com](https://towardsdatascience.com/social-media-sentiment-analysis-49b395771197)
