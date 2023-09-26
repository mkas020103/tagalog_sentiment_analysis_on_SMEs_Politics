#!/usr/bin/env python
# coding: utf-8

# # Import modules

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import wordcloud
import seaborn as sns
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.metrics import classification_report, confusion_matrix
from langid.langid import LanguageIdentifier, model
from sklearn.model_selection import GridSearchCV
import re
import nltk
from collections import Counter
from sklearn.model_selection import train_test_split
from nltk import bigrams
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import gensim
import gensim.corpora as corpora
import gensim.models.ldamodel
import pyLDAvis
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
import warnings
from itertools import combinations
from collections import Counter
import time
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
#from gensim.model import CoherenceModel
nltk.download('stopwords')

# modelling imports
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from transformers import AutoTokenizer, TFAutoModel
from gensim.models import Word2Vec
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
import joblib
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.models import load_model
from imblearn.over_sampling import SMOTE
import os


# In[2]:


# Suppress DeprecationWarning from Pillow
warnings.filterwarnings("ignore", category=DeprecationWarning)


# # Read data
# This class is use to: 
# 
#     -read the csv file on the given data and the filipino stopwords
#     -pre-clean the data by dropping unnecesary columns and null values.
#     -split the filipino text and english text
#     
#     The parameters are:
#         -filename (string format)
#         -stopwords (string format) (optional)

# # Preprocessing Text Data

# In[3]:


class preprocess:
    def __init__(self, df, comment_column: str, dups: int = 0, use_for: int = 1, dup_subset: list = None, added_stopwords = None, drop : list = None, sent_column: list = None):
        self.use_for = use_for
        self.columns_to_drop = drop
        self.sents = sent_column
        self.column = comment_column
        self.dups = dups
        self.dup_subset = dup_subset
        self.filipino_stopwords = []
        # read data
        if type(df) == str:
            self.df = pd.read_csv(df)
        elif type(df) == pd.core.frame.DataFrame:
            self.df = df
        else:
            print('Input a filename or the dataframe.')
            return
        
        # read added stopwords
        if added_stopwords != None:
            stopwords_df = pd.read_csv(added_stopwords)
            self.filipino_stopwords = [word for word in stopwords_df['.stopwords']]
        
        # print columns
        column_headers = list(self.df.columns.values)
        print("Column Headers: ", column_headers)
        self.stop_words = self.filipino_stopwords          # words to stop  
        self.tokenizer = RegexpTokenizer(r"\w+|[^\w\s]+")  # tokenizer
        self.substituted_text = []                         # empty list of to be substituted text
        self.initial_tokens = []                           # empty list of initial tokens
        self.stopped_tokens = []                           # empty list of tokens without the stopwords
        self.stopped_text = []                             # empty list of tokens in string format
        
        if self.sents is not None:
            self.sentiment_dict = {                            # switch name into integers
                'nagative': 0,
                'positive': 2,
                'negative': 0,
                'neutral': 1,
                'neitral': 1,
                'positve': 2,
                'posiive': 2,
                'neitral': 1,
                'posiive': 2,
                'Negative': 0,
                'Neutral': 1,
                'Positive': 2,
                'Postive': 2,
                'positve': 2,
                'neiutral': 1,
                'nuetral': 1,
                'neutrl': 1,
                'neutrsal': 1,
                '1':1,
                '2':2,
                '0':0,
                0.0: 0,
                1.0: 1,
                2.0: 2
            }
        
        # show raw dataframe
        print("shape of raw dataframe: ", self.df.shape)
        
        # if there are no sentiment columns (yet)
        if self.sents is not None:
            # Drop Null Values
            for sent_column in self.sents:
                self.df.dropna(subset=[sent_column], inplace=True)
                
            # Map the categories to numbers
            print("Sentiment null count: ", self.df[self.sents].isnull().sum())
            for sentiments_column in self.sents:
                self.df[sentiments_column] = self.df[sentiments_column].apply(lambda x: x.lower().strip() if isinstance(x, str) else x)
                print(self.df[self.sents].value_counts())
                print("shape of mapped dataframe: ", self.df.shape)

            for sentiments_column in self.sents:
                self.df[sentiments_column] = self.df[sentiments_column].apply(lambda x: x.lower() if isinstance(x, str) else x).map(self.sentiment_dict)
                print("Sentiment null count: ", self.df[self.sents].isnull().sum())
                
            if sentiments_column != 'sentiments':
                self.df['sentiments'] = self.df[sentiments_column]
                self.df.drop([sentiments_column], axis = 1, inplace = True) 
        
        # preprocess
        self.df.dropna(subset=[self.column], inplace=True)
        self.preprocessing_steps()
        
        # append to dataframe
        self.df["features"] = self.stopped_tokens
        self.df["features_string_format"] = self.stopped_text
        
        # drop null values in the comment
        self.df.dropna(subset=[self.column], inplace=True)
        print("shape of dataframe when null comments were dropped: ", self.df.shape)
        
        # drop duplicated values
        if self.dups == 1:
            if self.dup_subset == None:
                print("You need to input the subsets of the duplicate values.")
                return
            if not self.dup_subset:
                print("List is empty. Input columns in the 'dup_subset' variable.")
                return
            self.df.dropna(subset=self.dup_subset, inplace=True)
            self.df = self.df.drop_duplicates(subset=self.dup_subset)
            print("shape of dataframe when preprocessed and duplicated values where dropped: ", self.df.shape)
            
        # drop unnecesary columns
        if self.columns_to_drop != None:
            self.df.drop(self.columns_to_drop, axis = 1, inplace = True) 
        
        # drop rows that have no values (if for preprocessing data (1), else for inputting neutral sentiments (0))
        if self.use_for == 1:
            self.df.dropna(subset=["features_string_format"], inplace=True)
            print("shape of final dataframe when rows that have null values where dropped: ", self.df.shape)
        else:
            if self.sents != None:
                for columns in self.sents:
                    self.df[columns] = self.df[columns].fillna('neutral')

        
    def preprocessing_steps(self):
        # substitute tokens
        self.initial_texts = [self.substitute(text.lower()) for text in self.df[self.column]]
        
        # tokenize the text
        self.initial_tokens = [self.tokenizer.tokenize(text) for text in self.initial_texts]
        
        # remove stopwords
        self.stopped_tokens = [[token for token in token_list if len(token) >= 2 
                                and token not in self.stop_words] for token_list in self.initial_tokens]
        
        # make them in string format
        self.stopped_text = [' '.join(tokens) for tokens in self.stopped_tokens]
    
    def paraphrase(self, list_of_paraphrasing: list = None, category: int = None):
        if list_of_paraphrasing is None or category is None:
            print('Input the list of words to be paraphrased and the category value.')
            return
        
        temp_df = self.df[self.df[self.sents[0]] == category]

        if temp_df.empty:
            print(f"No rows found for category {category}.")
            return
        
        new_rows = []  # Initialize a list to store new rows

        for index, row in temp_df.iterrows():
            original_text = row['features_string_format']  # Initialize with original text
            paraphrased_text = original_text  # Initialize paraphrased_text

            for words in list_of_paraphrasing:
                original_word, paraphrased_word = words
                paraphrased_text = re.sub(r'\b{}\b'.format(re.escape(original_word)), paraphrased_word, paraphrased_text)

            if paraphrased_text != original_text:
                # If there's a paraphrased text, create a new row with it
                new_row = row.copy()
                new_row['features_string_format'] = paraphrased_text
                new_rows.append(new_row)

        # Create a new DataFrame using the list of new rows
        paraphrased_df = pd.DataFrame(new_rows, columns=temp_df.columns)

        # Return the new DataFrame
        return paraphrased_df
    
    def get_pattern_data(self,pattern=None,sent=None):
        if pattern==None:
            print('Input the word/s of the data you want to get')
            return
        elif type(pattern) != str:
            print('Input the word/s of the data you want to get')
            return
        expression = r'\b{}\b'.format(pattern)
        filtered_comments = self.df[self.df[self.column].str.contains(expression, case=False, regex=True)]
        
        if sent==None:
            print(1)
            return filtered_comments
        elif type(sent) != int:
            print('Choose only between 0 to 2.')
            return
        elif sent in [0,1,2]:
            print(sent)
            data = filtered_comments[filtered_comments[self.sents] == sent]
            print(filtered_comments)
            return data
        
    def substitute(self, text):
        text = re.sub(r'([?!])+', r'\1', text)
        text = re.sub(r'@[\w\.]+', '', text, flags=re.IGNORECASE)  # Remove TikTok tags starting with '@'
        text = re.sub(r'#\w+', '', text, flags=re.IGNORECASE)      # Remove hashtags starting with '#'
        text = re.sub(r'\S*@\S*', '', text, flags=re.IGNORECASE)   # Remove any remaining email-like patterns
        text = re.sub(r'\b(?:http\S+|@\S+)\b', '', text, flags=re.IGNORECASE)  # Remove URLs and remaining '@' patterns
        text = re.sub(r'[^a-zA-Z0-9\s\?\!\.\,]', '', text)  # Remove special characters except for punctuation
        text = text.replace('\n', ' ')        # Replace newline characters with spaces
        text = re.sub(r'[^a-zA-Z0-9\s\?\!\.\,]', '', text)  # Remove special characters again (duplicate line)
        text = re.sub(r'\b\w*haha\w*\b', 'hahahaha', text, flags=re.IGNORECASE)  # Replace variations of 'haha' with 'hahahaha'
        text = re.sub(r'bomata', 'bumata', text)
        text = re.sub(r'(\w)\1{2,}', r'\1', text)
        text = re.sub(r'bbo', 'bobo', text)
        text = re.sub(r'tnga', 'tanga', text)
        text = re.sub(r'pnget', 'panget', text)
        text = re.sub(r'thanks', 'thank', text)
        text = re.sub(r'\bpoh\b', 'po', text)
        text = re.sub(r'\sno\b', 'sino', text)
        text = re.sub(r'\bparents\b', 'parent', text)
        text = re.sub(r'\bpano\b', 'paano', text)
        text = re.sub(r'\bpanu\b', 'paano', text)
        text = re.sub(r'\bpno\b', 'paano', text)
        text = re.sub(r'\bpaanu\b', 'paano', text)
        text = re.sub(r'\bnman\b', 'naman', text)
        text = re.sub(r'\bnmn\b', 'naman', text)
        text = re.sub(r'\bkhit\b', 'kahit', text)
        text = re.sub(r'\bkht\b', 'kahit', text)
        text = re.sub(r'\bpwedi\b', 'pwede', text)
        text = re.sub(r'\bpwde\b', 'pwede', text)
        text = re.sub(r'\bpede\b', 'pwede', text)
        text = re.sub(r'\byong\b', 'yung', text)
        text = re.sub(r'\bkasu\b', 'kaso', text)
        text = re.sub(r'\bkso\b', 'kaso', text)
        text = re.sub(r'\bhndi\b', 'hindi', text)
        text = re.sub(r'\bd\b', 'hindi', text)
        text = re.sub(r'\bsan\b', 'saan', text)
        text = re.sub(r'\baplay\b', 'apply', text)
        text = re.sub(r'\baply\b', 'apply', text)
        text = re.sub(r'\bappy\b', 'apply', text)
        text = re.sub(r'ngaun', 'ngayon', text)
        text = re.sub(r'\bnla\b', 'nila', text)
        text = re.sub(r'nakha', 'nakuha', text)
        text = re.sub(r'\baqo\b', 'ako', text)
        text = re.sub(r'\baq\b', 'ako', text)
        text = re.sub(r'aqoeng', 'akong', text)
        text = re.sub(r'\bala\b', 'wala', text)
        text = re.sub(r'walang', 'wala', text)
        text = re.sub(r'\bwla\b', 'wala', text)
        text = re.sub(r'\bwalng\b', 'wala', text)
        text = re.sub(r'\bwlang\b', 'wala', text)
        text = re.sub(r'\bmam\b', 'maam', text)
        text = re.sub(r'\bbkit\b', 'bakit', text)
        text = re.sub(r'perent', 'parent', text)
        text = re.sub(r'\bpo b\b', 'po ba', text)
        text = re.sub(r'\bganun\b', 'ganoon', text)
        text = re.sub(r'\bganon\b', 'ganoon', text)
        text = re.sub(r'\bgnon\b', 'ganoon', text)
        text = re.sub(r'\bnde\b', 'hindi', text)
        text = re.sub(r'\bndi\b', 'hindi', text)
        text = re.sub(r'\bdi\b', 'hindi', text)
        text = re.sub(r'\bd\b', 'hindi', text)
        text = re.sub(r'\bcnbi\b', 'sinabi', text)
        text = re.sub(r'\bkyo\b', 'kayo', text)
        text = re.sub(r'\btyo\b', 'tayo', text)
        text = re.sub(r'\btau\b', 'tayo', text)
        text = re.sub(r'\bbat\b', 'bakit', text)
        text = re.sub(r'\bnakha\b', 'nakuha', text)
        text = re.sub(r'\bnkuha\b', 'nakuha', text)
        text = re.sub(r'\bnkha\b', 'nakuha', text)
        text = re.sub(r'\bmrami\b', 'marami', text)
        text = re.sub(r'\bmdami\b', 'marami', text)
        text = re.sub(r'\btlaga\b', 'talaga', text)
        text = re.sub(r'\btlaaga\b', 'talaga', text)
        text = re.sub(r'\bq\b', 'ako', text)
        text = re.sub(r'\bkaayu\b', 'kayo', text)
        text = re.sub(r'\d+', '', text)
        return text


# # Exploratory Data Analysis
# 
# This class is use to: 
# 
#     -explore the data
#     -find insights within the data 
#     -outputs the visualization that would be useful for preprocessing and presentation purposes.
# 
#     The following are the outputs:
#         -descriptions (i.e. categories, number of values, mean)
#         -wordcloud
#         -sentiment plots
#         -bar plots
#         

# In[31]:


class eda:
    def __init__(self, df, text_column_name: str, sent_column_name: str = None, folder: str = None, background_color: str = '#FFFFFF', text_color: str = '#000000', bar_color: str = '#121166'):
        self.text_column_name = text_column_name
        self.sent_column_name = sent_column_name
        self.folder = folder
        self.df = df
        self.wordcloud = WordCloud(width=1200, height=800, max_words=500, background_color="white", scale=2)
        self.all_bigrams = []
        self.all_text = ' '.join(self.df[self.text_column_name])
        self.background_color = background_color
        self.text_color = text_color
        self.bar_color = bar_color
        self.filtered_comments = None
        self.all_grams = []
        self.sentiment_dict = {                            # switch name into integers
                'negative': 0,
                'positive': 2,
                'neutral': 1,
            }
        if self.sent_column_name != None:
            # map the sentiments into 0,1,2
            self.df[self.sent_column_name] = self.df[self.sent_column_name].apply(lambda x: x.lower() if isinstance(x, str) else x).map(self.sentiment_dict)
            print(self.df[self.sent_column_name].value_counts())
        if self.folder != None:
            if not os.path.isdir(self.folder):
                print('File doesn\'t exist.')

    def descriptions(self):
        print('shape:\n', self.df.shape, '\n')
        if self.sent_column_name != None:
            print('sentiment count:\n', self.df[self.sent_column_name].value_counts(), '\n')
        print('columns:', list(self.df.columns.values), '\n')

    def word_number(self):
        word_counts = self.df[self.text_column_name].str.split().apply(len)
        bins = np.arange(1, max(word_counts) + 2)  # Start from 1 and end at max + 1
        plt.hist(word_counts, bins=bins)
        plt.xlabel('Number of Words')
        plt.ylabel('Frequency')
        plt.title('Number of Words in Comments')

        # Customize x-axis tick positions and labels
        tick_positions = np.arange(0, 30, 5)  # Start from 1, increment by 5, end at 30
        tick_labels = [str(pos) for pos in tick_positions]
        plt.xticks(tick_positions, tick_labels)

        plt.show()
        
    def pattern_matching(self, pattern, show_sents=None, show_comments=None):
        expression = r'\b{}\b'.format(pattern)  # Regular expression pattern to match the word or words
        
        self.filtered_comments = self.df[self.df[self.text_column_name].str.contains(expression, case=False, regex=True)]
        
        num_comments = len(self.filtered_comments)
        
        # print the total comments having the pattern
        print(f"Total number of comments containing '{pattern}' = {num_comments}:100%")
        
        # if user want to see the percentage of each sentiments
        if show_sents != None:
            # calculate the number of each sentiments
            num_negative_comments = len(self.filtered_comments[self.filtered_comments[self.sent_column_name] == 0])
            num_positive_comments = len(self.filtered_comments[self.filtered_comments[self.sent_column_name] == 2])
            num_neutral_comments = len(self.filtered_comments[self.filtered_comments[self.sent_column_name] == 1])
            
            negative_percentage = (num_negative_comments/num_comments) * 100
            positive_percentage = (num_positive_comments/num_comments) * 100
            neutral_percentage = (num_neutral_comments/num_comments) * 100
            
            print(f"Positive = {num_positive_comments}:{positive_percentage:.3f}%")
            print(f"Negative = {num_negative_comments}:{negative_percentage:.3f}%")
            print(f"Neutral = {num_neutral_comments}:{neutral_percentage:.3f}%")
            
        # if user want to see the comments
        if show_comments != None:
            print('All comments with the pattern: \n')
            for word in self.filtered_comments[self.text_column_name]:
                print(word)
        
    def top_words(self, top: int, sentiment: str):
        # Tokenize the text
        tokens = self.all_text.split()

        # Count the frequency of each word
        word_counts = pd.Series(tokens).value_counts()

        # Select the top 10 words
        top_10_words = word_counts.head(top)

        # Set the color specifications
        background_color = self.background_color
        text_color = self.text_color
        bar_color = self.bar_color

        # Create the bar chart
        fig, ax = plt.subplots(figsize=(11, 7), facecolor=background_color)
        ax.barh(top_10_words.index, top_10_words.values, color=bar_color)

        # Set the background and text colors
        fig.set_facecolor(background_color)
        ax.set_facecolor(background_color)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        ax.tick_params(axis='x', colors=text_color)
        ax.tick_params(axis='y', colors=text_color)

        # Add labels and title
        plt.xlabel('Frequency', color=text_color)
        plt.ylabel('Words', color=text_color)
        plt.title('Top {} {} Words'.format(top, sentiment), color=text_color)

        # Show the chart
        if self.folder != None:
            plt.savefig('{}/top_{}words_{}.png'.format(self.folder, top, self.folder))
        plt.show()
        
    def grams(self, top: int, sentiment: str, gram_type: str):
        # create an empty grams
        self.all_grams = []
        
        # Determine the gram size based on gram_type
        if gram_type == 'unigram':
            print(1)
            gram_size = 1
        elif gram_type == 'bigram':
            print(2)
            gram_size = 2
        elif gram_type == 'trigram':
            print(3)
            gram_size = 3
        else:
            raise ValueError("Invalid gram_type. Supported values are 'unigram', 'bigram', and 'trigram'.")

        x = 0
        # Iterate over each text and compute grams
        for text in self.df[self.text_column_name]:
            # Tokenize the text into individual words
            tokens = text.split()
            
            if x == 0:
                print("gram_size: ", gram_size)
            
            # Generate n-grams
            grams_tokens = list(combinations(tokens, gram_size))

            # Add the grams to the list
            self.all_grams.extend(grams_tokens)
            
            x += 1

        # Count the frequency of each gram
        gram_counts = Counter(self.all_grams)

        # Select the top n grams
        top_n_grams = gram_counts.most_common(top)

        # Extract the gram phrases and their frequencies
        gram_phrases, gram_frequencies = zip(*top_n_grams)

        # Convert the tuples to lists
        gram_phrases = [', '.join(phrase) for phrase in gram_phrases]
        gram_frequencies = list(gram_frequencies)

        # Create the bar chart
        fig, ax = plt.subplots(figsize=(10, 8), facecolor=self.background_color)
        ax.barh(gram_phrases, gram_frequencies, color=self.bar_color)

        # Set the background and text colors
        fig.set_facecolor(self.background_color)
        ax.set_facecolor(self.background_color)
        ax.xaxis.label.set_color(self.text_color)
        ax.yaxis.label.set_color(self.text_color)
        ax.tick_params(axis='x', colors=self.text_color)
        ax.tick_params(axis='y', colors=self.text_color)

        # Add labels and title
        plt.xlabel('Frequency', color=self.text_color)
        plt.ylabel(f'{gram_type.capitalize()}s', color=self.text_color)
        plt.title(f'Top {top} {sentiment} {gram_type.capitalize()}s', color=self.text_color)

        # Save chart
        if self.folder is not None:
            plt.savefig(f'{self.folder}/top_{top}_{gram_type}s_{self.folder}.png')
        else:
            plt.savefig(f'top_{top}_{gram_type}s_{self.folder}.png')

        # Show the chart
        plt.show()

    def generate_wordcloud(self, value: int = None):
        # if there is a column name given
        if value is not None:
            joined_comments = ''
            # combine all the text into one string
            for index, row in self.df.iterrows():
                if row[self.sent_column_name] == value:
                    joined_comments += ' ' + str(row[self.text_column_name])
            # create cloud
            self.wordcloud.generate(joined_comments)
        else:
            joined_comments = ' '.join(self.df[self.text_column_name])

        # Split words
        words = joined_comments.split()

        # Count word frequencies
        word_counts = Counter(words)

        # Get the top 10 words based on frequency
        top_words = [word for word, count in word_counts.most_common(300)]

        # Generate word cloud from the top 10 words
        self.wordcloud.generate(' '.join(top_words))
        
        # Set the background color
        background_color = self.background_color
        self.wordcloud.background_color = background_color

        # Set the colors of the words
        if value == None:
            word_colors = ['#331D2C', '#3F2E3E', '#3C2A21', '#400E32']
        else:
            # if negative
            if value == 0:
                word_colors = ['#900C3F', '#C70039', '#F94C10', '#FE0000']
            # if neutral
            elif value == 1:
                word_colors = ['#61677A', '#0E2954', '#435B66', '#394867']
            # if positive
            elif value == 2:
                word_colors = ['#285430', '#54B435', '#379237', '#82CD47']
                
        self.wordcloud.recolor(color_func=lambda *args, **kwargs: random.choice(word_colors))

        if self.folder != None:
            if value is not None:
                if value == 0:
                    self.wordcloud.to_file('{}/{}_negative_wordcloud.png'.format(self.folder, self.folder))
                elif value == 1:
                    self.wordcloud.to_file('{}/{}_neutral_wordcloud.png'.format(self.folder, self.folder))
                elif value == 2:
                    self.wordcloud.to_file('{}/{}_positive_wordcloud.png'.format(self.folder, self.folder))
            else:
                self.wordcloud.to_file('{}/{}_wordcloud.png'.format(self.folder, self.folder))
        else:
            if value is not None:
                if value == 0:
                    self.wordcloud.to_file('{}_negative_wordcloud.png'.format(self.folder))
                elif value == 1:
                    self.wordcloud.to_file('{}_neutral_wordcloud.png'.format(self.folder))
                elif value == 2:
                    self.wordcloud.to_file('{}_positive_wordcloud.png'.format(self.folder))
            else:
                self.wordcloud.to_file('{}_wordcloud.png'.format(self.folder))
            
        # show plot
        plt.imshow(self.wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        
    def pie_sentiment(self):
        value_counts = self.df[self.sent_column_name].value_counts()

        labels = value_counts.index
        counts = value_counts.values

        colors = ['#001C30', '#176B87', '#64CCC5']
        background_color = self.background_color

        # Map the index values to the desired labels
        sentiment_labels = labels.map({0.0: 'negative', 1.0: 'neutral', 2.0: 'positive'})

        plt.figure(facecolor=background_color)
        plt.style.use('default')
        plt.pie(counts, labels=sentiment_labels, autopct='%1.1f%%', colors=colors)
        plt.axis('equal')
        plt.gca().set_facecolor(background_color)
        if self.folder != None:
            plt.savefig('{}/{}_pie_plot_sentiment.png'.format(self.folder, self.folder))
        else:
            plt.savefig('{}_pie_plot_sentiment.png'.format(self.folder))
        plt.show()

        
    def bar_sentiment(self):
        # Set the color specifications
        background_color = self.background_color
        text_color = self.text_color
        bar_color = self.bar_color

        # Set the background color
        plt.figure(facecolor=background_color)

        # Count the occurrences of each sentiment
        sentiment_counts = self.df[self.sent_column_name].value_counts()

        # Define the color specifications
        color_mapping = {'negative': '#D21312', 'positive': '#03C988', 'neutral': '#576CBC'}

        # Map the index values to the desired labels
        sentiment_labels = sentiment_counts.index.map({0.0: 'negative', 1.0: 'neutral', 2.0: 'positive'})

        # Create a bar plot with custom colors and updated labels
        plt.bar(sentiment_labels, sentiment_counts.values, color=[color_mapping.get(sentiment, bar_color) for sentiment in sentiment_labels])

        # Set the text color
        plt.xlabel('Sentiments', color=text_color)
        plt.ylabel('Count', color=text_color)

        # Set the title and text colors based on column_name
        plt.title('Sentiments', color=text_color)

        # Set the tick colors
        plt.tick_params(colors=text_color)

        # Set the plot area color to the same as the background color
        plt.gca().set_facecolor(background_color)

        # Display the plot
        if self.folder != None:
            plt.savefig('{}/{}_bar_sentiment_plot.png'.format(self.folder, self.folder))
        else:
            plt.savefig('{}_bar_sentiment_plot.png'.format(self.folder))
        plt.show()


# # Sentiment Analysis
# 
#     this class is designed to use a pre-trained model to determine wether a text is positive or negative

# In[7]:


class sentiment_analysis:
    def __init__(self, df):
        self.df = df
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.fasttext_model = fasttext.load_model('C:\\project\\downloads\\lid.176.ftz')
        
        self.do_sentiments()
        
    def do_sentiments(self):
        # declare a list of sentiments
        english_sentiments = []
        
        for index, row in self.df.iterrows():
            if row['language'] == "English":
                # Perform sentiment analysis with VADER
                vader_sentiment = self.vader_analyzer.polarity_scores(row['features_string_format'])
                # Get the most probable sentiment label
                sentiment_label = max(vader_sentiment, key=vader_sentiment.get)
                # Map the sentiment labels to custom labels
                if sentiment_label == "neu":
                    sentiment_label = "neutral"
                elif sentiment_label == "pos" or sentiment_label == "compound":
                    sentiment_label = "positive"
                else:
                    sentiment_label = "negative"
                # Update the dataframe with sentiment label
                english_sentiments.append(sentiment_label)
            else:
                english_sentiments.append(None)
        
        # append to dataframe
        self.df['english_sents'] = english_sentiments
        
    def combine_sentiments(self):
        english_tagalog_sents = []
        for index, row in self.df.iterrows():
            if row['language'] == 'English':
                english_tagalog_sents.append(row['english_sents'])
            else:
                english_tagalog_sents.append(row['Comment type'])
        self.df['sentiments'] = english_tagalog_sents
            


# # Unsupervised using LDA

# In[8]:


from gensim import corpora, models
import pyLDAvis.gensim_models

class LDA:
    def __init__(self, df):
        self.df = df
        self.lowered_token_list = []
        self.corpus = []
        self.model = None
        self.modelling()
        
    def preparing(self):
        # lowercase the tokens
        self.lowered_token_list = []
        
        # put the values into the empty list
        for index, row in self.df.iterrows():
            lowercase_tokens = []
            for token in row['features']:
                lowercase_tokens.append(token.lower())
            self.lowered_token_list.append(lowercase_tokens)
        
        # initialize id2word here
        self.id2word = corpora.Dictionary(self.lowered_token_list)
        
        # put the tokens into the corpus
        for list_of_tokens in self.lowered_token_list:
            new = self.id2word.doc2bow(list_of_tokens)
            self.corpus.append(new)
    
    def modelling(self):
        self.preparing()
        self.model = models.LdaModel(
            corpus=self.corpus,
            id2word=self.id2word,
            num_topics=3,
            random_state=42,
            update_every=1,
            passes=10,
            chunksize=180,
            alpha='auto'
        )
    
    def output(self):
        pyLDAvis.enable_notebook()
        vis = pyLDAvis.gensim_models.prepare(self.model, self.corpus, self.id2word, mds='mmds', R=30)
        pyLDAvis.save_html(vis, 'visualization.html')  # Save the visualization to an HTML file


# # Unsupervised using K-means

# In[9]:


class KMeansClustering:
    def __init__(self, df):
        self.df = df
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(df['features_string_format'])
        
        self.min_clusters = 1
        self.max_clusters = 6
        self.cluster_range = range(self.min_clusters, self.max_clusters)
        self.inertia_values = []
        self.silhouette_scores = []
        
        self.find_best_cluster()
        self.plot()
        
    def find_best_cluster(self):
        for num_clusters in self.cluster_range:
            kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
            kmeans.fit(self.tfidf_matrix)
            self.inertia_values.append(kmeans.inertia_)

            labels = kmeans.labels_
            num_unique_labels = len(np.unique(labels))

            if num_unique_labels > 1:
                score = silhouette_score(self.tfidf_matrix, labels)
                self.silhouette_scores.append(score)
            else:
                self.silhouette_scores.append(0.0)
        
    def plot(self):
        if not self.cluster_range or not self.inertia_values or not self.silhouette_scores:
            print("Insufficient data to plot.")
            return
    
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(self.cluster_range, self.inertia_values, marker='o', color='green')
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Curve')

        ax2 = ax1.twinx()
        ax2.plot(self.cluster_range, self.silhouette_scores, marker='o', color='red')
        ax2.set_ylabel('Silhouette Score')

        plt.show()
        
    def update_database(self, best_cluster):
        # cluster
        kmeans = KMeans(n_clusters=best_cluster, init='k-means++', random_state=42)
        kmeans.fit(self.tfidf_matrix)
        
        # update dataframe
        self.df['cluster_label'] = kmeans.labels_.astype(str)
        


# In[1]:


class models:
    def __init__(self, df=None, xtrain=None, xtest=None, ytrain=None, ytest=None, x: str=None, y: str=None, filename: str=None):
        self.filename = filename
        self.x = x
        self.y = y
        self.df = df
        if not self.df.empty:
            if self.x == None:
                print('Input the column name of x data.')
                return
            if self.y == None:
                print('Input the column name of y data.')
                return
            if self.filename == None:
                print('Input the filename of the data.')
                return
            self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(self.df[self.x], self.df[self.y], 
                                                                                test_size=.20, stratify=df[self.y])
        elif self.df.empty:
            self.xtrain = xtrain 
            self.xtest = xtest
            self.ytrain = ytrain
            self.ytest = ytest
            if xtrain.empty:
                print('Input the xtrain.')
                return
            if xtest.empty:
                print('Input the xtest.')
                return
            if ytrain.empty:
                print('Input the ytrain.')
                return
            if ytest.empty:
                print('Input the ytest.')
                return
        else:
            print('Input a dataframe or the training and testing data.')
            return
        self.model = None
        self.vectorizer = TfidfVectorizer()
        self.xtrain_vectors = self.vectorizer.fit_transform(self.xtrain)
        self.xtest_vectors = self.vectorizer.transform(self.xtest)
        self.ytest = self.ytest.astype(int)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.ytrain)
        self.model_name = None
        self.y_predicted = None
        
    def bert_classification(self, epochs: int):
        self.model_name = 'bert'
        # Model framework
        e_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4'
        p_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
        
        bert_preprocess = hub.KerasLayer(p_url)
        bert_encoder = hub.KerasLayer(e_url)
        
        article = tf.keras.layers.Input(shape=(), dtype=tf.string, name='article')
        p_text = bert_preprocess(article)
        output = bert_encoder(p_text)
        l = tf.keras.layers.BatchNormalization()(output['pooled_output'])
        l = tf.keras.layers.Dropout(0.1, name='dropout_1')(l)
        l = tf.keras.layers.Dense(128, activation='relu')(l)
        l = tf.keras.layers.Dropout(0.1, name='dropout')(l)
        l = tf.keras.layers.Dense(3, activation='softmax', name='output')(l)

        self.model = tf.keras.Model(inputs=[article], outputs=[l])
        self.model.summary()
        
        self.model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
                           loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Start the time checker of the program
        start_time = time.time()
        
        # Train the model
        self.model.fit(self.xtrain, self.ytrain, epochs=epochs)
        
        # End time
        end_time = time.time()
        
        # Calculate the elapsed time
        elapsed_time = end_time - start_time

        # Print the elapsed time
        print('Elapsed time:', elapsed_time / 60)
        
        self.model.save('bert_model_{}.h5'.format(self.filename))
        
    def svm(self):
        self.model_name = 'svm'
        # Define the hyperparameter grid
        param_grid = {
            'C': [0.1, 1, 10, 100],  # Penalty parameter C
            'kernel': ['linear', 'rbf', 'poly'],  # Kernel type
            'gamma': ['scale', 'auto']  # Kernel coefficient for 'rbf' and 'poly' kernels
        }
        clf_svm = svm.SVC(kernel='linear')
        
        # Instantiate GridSearchCV
        grid_search = GridSearchCV(clf_svm, param_grid, cv=10)  # 5-fold cross-validation
        
        # Fit the gridsearch to the data
        grid_search.fit(self.xtrain_vectors, self.ytrain)
        
        # Access the best parameters
        best_params = grid_search.best_params_
        
        # Create a new SVM model with the best parameters
        best_svm_model = svm.SVC(**best_params)
    
        # Train the best SVM model on the entire training data
        best_svm_model.fit(self.xtrain_vectors, self.ytrain)

        # save model to class
        self.model = best_svm_model
        
        # Save the model to a file
        joblib.dump(best_svm_model, 'svm_model_{}.pkl'.format(self.filename))

        # Save the vectorizer
        joblib.dump(self.vectorizer, 'svm_vectorizer_{}.pkl'.format(self.filename))

        # Save the label encoder
        joblib.dump(self.label_encoder, 'svm_label_encoder_{}.pkl'.format(self.filename))
        
    def decision_tree(self):
        self.model_name = 'dectree'
        # Define the hyperparameter grid
        param_grid = {
            'criterion': ['gini', 'entropy'],         # Split criterion
            'splitter': ['best', 'random'],          # Strategy for choosing split
            'max_depth': [None, 10, 20, 30],         # Maximum depth of the tree
            'min_samples_split': [2, 5, 10],         # Minimum number of samples required to split an internal node
            'min_samples_leaf': [1, 2, 4],           # Minimum number of samples required to be at a leaf node
            'max_features': ['sqrt', 'log2'] # Number of features to consider for the best split
        }

        # Create the Decision Tree model
        dt_model = DecisionTreeClassifier()

        # Instantiate GridSearchCV
        grid_search = GridSearchCV(dt_model, param_grid, cv=10)  # 5-fold cross-validation

        # Fit GridSearchCV to the data
        grid_search.fit(self.xtrain_vectors, self.ytrain)

        # Access the best parameters
        best_params = grid_search.best_params_

        # Create a new Decision Tree model with the best parameters
        best_dt_model = DecisionTreeClassifier(**best_params)

        # Train the best Decision Tree model on the entire training data
        best_dt_model.fit(self.xtrain_vectors, self.ytrain)
        
        self.model = best_dt_model
        
        # Save the model to a file
        joblib.dump(best_dt_model, 'dectree_model_{}.pkl'.format(self.filename))

        # Save the vectorizer
        joblib.dump(self.vectorizer, 'dectree_vectorizer_{}.pkl'.format(self.filename))

        # Save the label encoder
        joblib.dump(self.label_encoder, 'dectree_encoder_{}.pkl'.format(self.filename))
        
    def random_forest(self):
        self.model_name = 'rf'
        # Define the hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],          # Number of trees in the forest
            'criterion': ['gini', 'entropy'],         # Split criterion for individual trees
            'max_depth': [None, 10, 20, 30],         # Maximum depth of the individual trees
            'min_samples_split': [2, 5, 10],         # Minimum number of samples required to split an internal node
            'min_samples_leaf': [1, 2, 4],           # Minimum number of samples required to be at a leaf node
            'max_features': ['sqrt', 'log2'] # Number of features to consider for the best split
        }

        # Create the Random Forest model
        rf_model = RandomForestClassifier()

        # Instantiate GridSearchCV
        grid_search = GridSearchCV(rf_model, param_grid, cv=5)  # 5-fold cross-validation

        # Fit GridSearchCV to the data
        grid_search.fit(self.xtrain_vectors, self.ytrain)

        # Access the best parameters
        best_params = grid_search.best_params_

        # Create a new Random Forest model with the best parameters
        best_rf_model = RandomForestClassifier(**best_params)

        # Train the best Random Forest model on the entire training data
        best_rf_model.fit(self.xtrain_vectors, self.ytrain)
        
        self.model = best_rf_model
        
        # Save the model to a file
        joblib.dump(best_rf_model, 'dectree_model_{}.pkl'.format(self.filename))

        # Save the vectorizer
        joblib.dump(self.vectorizer, 'dectree_vectorizer_{}.pkl'.format(self.filename))

        # Save the label encoder
        joblib.dump(self.label_encoder, 'dectree_encoder_{}.pkl'.format(self.filename))


        
    def evaluate_model(self):
        # if there are no models
        if self.model_name is None:
            print('You need to train a model first')
            return
        
        # if a bert model was used
        if self.model_name == 'bert':
            print('bert')
            prediction = self.model.predict(self.xtest)
            target_names = ['1.0', '2.0', '0']
            y_preds = np.argmax(prediction, axis=1)
            print("Classification Report: \n", classification_report(self.ytest, y_preds, target_names=target_names))
        # else if a svm or decision tree model was used
        elif self.model_name in ["svm", "dectree", "rf"]:
            print('svm or dectree')
            predictions = self.model.predict(self.xtest_vectors)
            accuracy = accuracy_score(self.ytest, predictions)
            f1_micro = f1_score(self.ytest, predictions, average='micro')
            f1_macro = f1_score(self.ytest, predictions, average='macro')
            classification_rep = classification_report(self.ytest, predictions)
            
            # Print scores and classification report
            print("Accuracy:", accuracy)
            print("F1 score (micro):", f1_micro)
            print("F1 score (macro):", f1_macro)
            print("Classification Report:")
            print(classification_rep)
            
            # Plot confusion matrix
            cm = confusion_matrix(self.ytest, predictions)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Labels')
            plt.ylabel('Actual Labels')
            plt.show()
        else:
            print('The models that can be used are "bert", "svm", "dectree", "rf". Choose one.')
            return


# # end of classes
