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

# In[3]:


class base:
    def __init__(self, df: str, added_stopwords:str = None):
        self.stop_words_filename = added_stopwords
        self.df_filename = df
        self.df = None
        self.stopwords_df = None
        self.stopwords = []
        
        # read file
        self.reading()
        
    def reading(self):
        # read stopwords csv
        if stopwords is not None:
            self.stopwords_df = pd.read_csv(self.stop_words_filename)
            self.stopwords = [word for word in self.stopwords_df['.stopwords']]
        else:
            self.stopwords_df = None
        # read csv file
        self.df = pd.read_csv(self.df_filename)


# # Preprocessing Text Data

# In[4]:


class preprocess(base):
    def __init__(self, df, added_stopwords, comment_column: str, dups: int = 0, drop : list = None, sent_column: list = None):
        self.columns_to_drop = drop
        self.sents = sent_column
        self.column = comment_column
        self.dups = dups
        super().__init__(df, added_stopwords)
        self.stop_words = stopwords.words('english')       # words to stop
        self.stopwords.extend(self.stop_words)             # combine all the stopwords
        self.tokenizer = RegexpTokenizer(r"\w+|[^\w\s]+")  # tokenizer
        self.substituted_text = []                         # empty list of to be substituted text
        self.initial_tokens = []                           # empty list of initial tokens
        self.stopped_tokens = []                           # empty list of tokens without the stopwords
        self.stopped_text = []                             # empty list of tokens in string format
        
        # print columns
        column_headers = list(self.df.columns.values)
        print("Column Headers: ", column_headers)
        if self.sents is not None:
            self.sentiment_dict = {                            # switch name into integers
                'nagative': 0,
                'positive': 2,
                'negative': 0,
                'neutral': 1,
                'positve': 2,
                'neitral': 1,
                'posiive': 2,
                'Negative': 0,
                'Neutral': 1,
                'Positive': 2,
                'Postive': 2,
                'positve': 2,
                0.0: 0,
                1.0: 1,
                2.0: 2,


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
                self.df[sentiments_column] = self.df[sentiments_column].str.lower()
                print(self.df[self.sents].value_counts())
                print("shape of mapped dataframe: ", self.df.shape)

            for sentiments_column in self.sents:
                self.df[sentiments_column] = self.df[sentiments_column].str.lower().map(self.sentiment_dict)
                print("Sentiment null count: ", self.df[self.sents].isnull().sum())
        
        # drop unnecesary columns
        if self.columns_to_drop != None:
            self.df.drop(self.columns_to_drop, axis = 1, inplace = True) 
        
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
            self.df = self.df.drop_duplicates(subset = "features_string_format")
            print("shape of dataframe when preprocessed and duplicated values where dropped: ", self.df.shape)
        
        # drop rows that have no values
        self.df.drop(self.df[self.df['features_string_format'] == ''].index, inplace=True)
        print("shape of final dataframe when rows that have null values where dropped: ", self.df.shape)

        
    def preprocessing_steps(self):
        # substitute the unnecesary text to ""
        self.substituted_text = [self.substitute(text).lower() for text in self.df[self.column]]
        
        # tokenize the text
        self.initial_tokens = [self.tokenizer.tokenize(text) for text in self.substituted_text]
        
        # remove stopwords
        self.stopped_tokens = [[token for token in token_list if len(token) > 2 and token not in self.stopwords] for token_list in self.initial_tokens]
        
        # make them in string format
        self.stopped_text = [' '.join(tokens) for tokens in self.stopped_tokens]
        
        
    def substitute(self, text):
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'\S*@\S*', '', text)
        text = re.sub(r'\b(?:http\S+|@\S+)\b', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s\?\!\.\,]', '', text)
        text = text.replace('\n', ' ')
        text = re.sub(r'[^a-zA-Z0-9\s\?\!\.\,]', '', text)
        text = re.sub(r'\b\w*haha\w*\b', 'hahahaha', text)
        text = re.sub(r'bomata', 'bumata', text)
        text = re.sub(r'(\w)\1{2,}', r'\1', text)
        text = re.sub(r'bbo', 'bobo', text)
        text = re.sub(r'tnga', 'tanga', text)
        text = re.sub(r'pnget', 'panget', text)
        text = re.sub(r'thanks', 'thank', text)

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

# In[5]:


class eda:
    def __init__(self, df, text_column_name: str, sent_column_name: str, folder: str = None):
        self.text_column_name = text_column_name
        self.sent_column_name = sent_column_name
        self.folder = folder
        self.df = df
        self.wordcloud = WordCloud(width=1200, height=800, max_words=500, background_color="white", scale=2)
        self.all_bigrams = []
        self.all_text = ' '.join(self.df[self.text_column_name])
        self.background_color = '#adadadff'
        self.text_color = '#000000'
        self.bar_color = '#121166'
        if self.folder != None:
            if not os.path.isdir(self.folder):
                print('File doesn\'t exist.')

    def descriptions(self):
        print('shape:\n', self.df.shape, '\n')
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
        
    def top_words(self, top: int, sentiment: str, filename: str):
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
            plt.savefig('{}/top_{}words_{}.png'.format(self.folder, top, filename))
        else:
            plt.savefig('top_{}words_{}.png'.format(top, filename))
        plt.show()
        
    def bigrams(self, top: int, sentiment: str, filename: str):
        # Iterate over each text and compute bigrams
        for text in self.df[self.text_column_name]:
            # Tokenize the text into individual words
            tokens = text.split()

            # Generate bigrams
            bigram_tokens = list(combinations(tokens, 2))

            # Add the bigrams to the list
            self.all_bigrams.extend(bigram_tokens)

        # Count the frequency of each bigram
        bigram_counts = Counter(self.all_bigrams)

        # Select the top 10 bigrams
        top_10_bigrams = bigram_counts.most_common(top)

        # Extract the bigram phrases and their frequencies
        bigram_phrases, bigram_frequencies = zip(*top_10_bigrams)

        # Convert the tuples to lists
        bigram_phrases = [', '.join(phrase) for phrase in bigram_phrases]
        bigram_frequencies = list(bigram_frequencies)

        # Set the color specifications
        background_color = self.background_color
        text_color = self.text_color
        bar_color = self.bar_color

        # Create the bar chart
        fig, ax = plt.subplots(figsize=(10, 8), facecolor=background_color)
        ax.barh(bigram_phrases, bigram_frequencies, color=bar_color)

        # Set the background and text colors
        fig.set_facecolor(background_color)
        ax.set_facecolor(background_color)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        ax.tick_params(axis='x', colors=text_color)
        ax.tick_params(axis='y', colors=text_color)

        # Add labels and title
        plt.xlabel('Frequency', color=text_color)
        plt.ylabel('Bigrams', color=text_color)
        plt.title('Top {} {} Bigrams'.format(top, sentiment), color=text_color)

        # Save chart
        if self.folder != None:
            plt.savefig('{}/top_{}bigrams_{}.png'.format(self.folder, top, filename))
        else:
            plt.savefig('top_{}bigrams_{}.png'.format(top, filename))
            
        # Show the chart
        plt.show()
        
    def english_tagalog(self):
        # Count the occurrences of each language
        language_counts = df['language'].value_counts()
        
        # Plotting the bar graph
        plt.bar(language_counts.index, language_counts.values)

        # Adding labels and title
        plt.xlabel('Language')
        plt.ylabel('Count')
        plt.title('Language Distribution')

        # Display the plot
        plt.show()

    def generate_wordcloud(self, filename: str, category: str = None, value: int = None):
        if category is not None:
            if value is None:
                print("Input category value.")
                return
            joined_comments = ''
            for index, row in self.df.iterrows():
                if row[category] == value:
                    joined_comments += ' ' + str(row[self.text_column_name])
            self.wordcloud.generate(joined_comments)
        else:
            joined_comments = ' '.join(self.df[self.text_column_name])
            self.wordcloud.generate(joined_comments)

        # Set the background color
        background_color = self.background_color
        self.wordcloud.background_color = background_color

        # Set the colors of the words
        word_colors = ['#b31212ff', '#121cb3ff', '#a51064ff', '#14630dff']
        self.wordcloud.recolor(color_func=lambda *args, **kwargs: random.choice(word_colors))

        if self.folder != None:
            self.wordcloud.to_file('{}/{}_wordcloud.png'.format(self.folder, filename))
        else:
            self.wordcloud.to_file('{}_wordcloud.png'.format(filename))
        plt.imshow(self.wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        
    def pie_sentiment(self, filename, column_name):
        value_counts = self.df[column_name].value_counts()

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
            plt.savefig('{}/{}_pie_plot_sentiment.png'.format(self.folder, filename))
        else:
            plt.savefig('{}_pie_plot_sentiment.png'.format(filename))
        plt.show()

        
    def bar_sentiment(self, filename: str, column_name: str):
        # Set the color specifications
        background_color = self.background_color
        text_color = self.text_color
        bar_color = self.bar_color

        # Set the background color
        plt.figure(facecolor=background_color)

        # Count the occurrences of each sentiment
        sentiment_counts = self.df[column_name].value_counts()

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
        if column_name == 'Comment type':
            plt.title('Given Sentiments', color=text_color)
        elif column_name == 'sentiments':
            plt.title('Tagalog & English Sentiments', color=text_color)
        elif column_name == 'english_sents':
            plt.title('English Sentiments', color=text_color)

        # Set the tick colors
        plt.tick_params(colors=text_color)

        # Set the plot area color to the same as the background color
        plt.gca().set_facecolor(background_color)

        # Display the plot
        if self.folder != None:
            plt.savefig('{}/{}_bar_sentiment_plot.png'.format(self.folder, filename))
        else:
            plt.savefig('{}_bar_sentiment_plot.png'.format(filename))
        plt.show()


# # Sentiment Analysis
# 
#     this class is designed to use a pre-trained model to determine wether a text is positive or negative

# In[6]:


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

# In[7]:


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

# In[8]:


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
        


# In[9]:


class models:
    def __init__(self, df, x: str, y: str, filename: str):
        self.filename = filename
        self.df = df
        self.x = x
        self.y = y
        self.model = None
        self.vectorizer = TfidfVectorizer()
        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(self.df[self.x], self.df[self.y], 
                                                                            test_size=.20, stratify=df[self.y])
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
        joblib.dump(best_svm_model, 'dectree_model_{}.pkl'.format(self.filename))

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
        joblib.dump(self.model, 'random_forest_model_{}.pkl'.format(self.filename))
        
        # Save the vectors
        self.xtrain.to_csv('{}_{}.csv'.format(self.model_name, self.filename))
        
        # Save the label encoder
        joblib.dump(self.label_encoder, 'random_forest_label_encoder_{}.pkl'.format(self.filename))


        
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
