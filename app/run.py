# basic libraries:
import json
import random
import numpy as np
import pandas as pd

# database connection libraries:
from sqlalchemy import create_engine

# string libraries:
import re
import string

# nlp libraries:
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from collections import Counter

# visualization libraries:
import plotly
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Scatter

from sklearn.externals import joblib

app = Flask(__name__)


# replacing the original function to one of my own:
"""
def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
"""

# same function as the one from the train_classifier.py script:
def tokenize(text):
    """
    Function to tokenize some text (divide into different words - 'tokens')
    
    Parameters:
    text (str): text to be tokenized
    
    Returns:
    list: tokens for the text passed as parameter 
    """
    
    # remove numbers:
    text = re.sub(r"[0-9]", " ", text)
    
    # convert to lowercase:
    text = text.lower()
    
    # remove punctuation characters:
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # divide into words:
    tokenized_words = word_tokenize(text)
    
    # remove stopwords:
    tokenized_words = [word for word in tokenized_words if word not in stopwords.words("english")]
    
    # reduce words to their root form:
    lemmed_words = [WordNetLemmatizer().lemmatize(w) for w in tokenized_words]
    
    return tokenized_words


# load data:
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterMessages', engine)

# load model:
model = joblib.load("../models/classifier.pkl")

# download of stopwords:
nltk.download('stopwords')


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals:
    df_genre = df.groupby(by=['genre'], as_index=False).agg({
        'message': ['count']
    })
    df_genre.columns = df_genre.columns.droplevel(0)
    df_genre.columns = ['genre', 'num_msgs']
    
    df_genre = df_genre.sort_values(by='num_msgs', ascending=True).head(10)
    
    # taking just a few words to try to plot in a Word Cloud (TOP 100):
    words = ' '.join(df['message'])
    tokens = tokenize(words)
    counts = Counter(tokens)
    sorted_dict = dict( sorted(counts.items(), key=lambda item: item[1], reverse=True) )
    frequency = list(sorted_dict.values())[:100]
    words = list(sorted_dict.keys())[:100]
    weights = [np.sqrt(random.random()*freq) for freq in frequency]
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

    # create visuals:
    graphs = [
        # chart #1:
        {
            'data': [
                Bar(
                    orientation='h',
                    x=df_genre['num_msgs'],
                    y=df_genre['genre']
                )
            ],

            'layout': {
                'title': 'TOP genres with more messages',
                'yaxis': {
                    'title': "Genre"
                },
                'xaxis': {
                    'title': "# of messages"
                }
            }
        },  
        
        # chart #2:
        {
            'data': [
                Scatter(
                    x=random.choices(range(len(words)), k=len(words)),
                    y=random.choices(range(len(words)), k=len(words)),
                    mode='text',
                    text=words,
                    #hovertext=['{0}{1}'.format(w, f) for w, f in zip(words, frequency)],
                    #hoverinfo='text',
                    textfont={
                        'size': weights, 
                        'color': colors
                    }
                )
            ],

            'layout': {     
                'title': 'TOP words used in the messages',
                'yaxis': {
                    'showgrid': False, 
                    'showticklabels': False,
                    'zeroline': False
                },
                'xaxis': {
                    'showgrid': False, 
                    'showticklabels': False,
                    'zeroline': False
                }
            }
        }
        
    ]
    
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
