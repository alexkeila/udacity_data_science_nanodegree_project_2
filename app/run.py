import json
import plotly
import numpy as np
import pandas as pd
import random

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from collections import Counter

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Scatter
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

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

def tokenize(text):
    # Remove numbers:
    text = re.sub(r"[0-9]", " ", text)
    
    # Convert to lowercase:
    text = text.lower()
    
    # Remove punctuation characters:
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    tokenized_words = word_tokenize(text)
    tokenized_words = [word for word in tokenized_words if word not in stopwords.words("english")]
    
    return tokenized_words

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterMessages', engine)

# load model
model = joblib.load("../models/classifier.pkl")

nltk.download('stopwords')


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    df_genre = df.groupby(by=['genre'], as_index=False).agg({
        'message': ['count']
    })
    df_genre.columns = df_genre.columns.droplevel(0)
    df_genre.columns = ['genre', 'num_msgs']
    
    df_genre = df_genre.sort_values(by='num_msgs', ascending=True).head(10)
    
    words = ' '.join(df['message'])
    tokens = tokenize(words)
    counts = Counter(tokens)
    
    sorted_dict = dict( sorted(counts.items(), key=lambda item: item[1], reverse=True) )
    
    words = list(sorted_dict.keys())[:100]
    frequency = list(sorted_dict.values())[:100]
    
    length = len(words)
    #colors = [plotly.colors.DEFAULT_PLOTLY_COLORS[random.randrange(1, 10)] for i in range(30)]
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
    weights = [np.sqrt(random.random()*freq) for freq in frequency]

    # create visuals
    graphs = [
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
        
        {
            'data': [
                Scatter(
                    x=random.choices(range(length), k=length),
                    y=random.choices(range(length), k=length),
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
