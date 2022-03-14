# basic libraries:
import sys
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
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

# machine learning libraries:
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import metrics
import pickle


def load_data(database_filepath):
    """
    Function to read the dataset from the database
    
    Parameters:
    database_filepath (str): path to where the database is located on disk
    
    Returns:
    Series: all messages named as X (input of a model) 
    dataframe: classification of all categories named as y (label of a model)
    list: name of the categories
    """
    
    
    # load data from database:
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterMessages', con=engine)
    
    # I do not know why, but for some reason, there are some observations with the related category equals 2?!?
    # forcing the observations with related = 2 to be = 1,
    # since it is a multilabel classification, but not a multi-output classification (each category is binary):
    # 13/03/2022 - Moved this code to the 'process_data.py' script
    ##df['related'] = np.where(df['related'] == 2, 1, df['related'])
    
    X = df['message']
    y = df.drop(['message', 'original', 'id', 'genre'], axis=1, inplace=False)
    
    category_names = y.columns
    
    return X, y, category_names


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
    
    return lemmed_words


def build_model():
    
    # create the pipeline object:
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # dictionary of hyperparameters to try and choose the best ones:
    # it takes a long long time to run with lots of combinations, so I tried to leave here just some few options:
    parameters = {
        'vect__max_features': (500, 750),
        #'clf__estimator__n_estimators': [20, 30], # I left the default value for n_estimators
        'clf__estimator': [
            RandomForestClassifier(),
            RidgeClassifier()
        ]
    }

    # after the cross validation, the model with the best parameters will be trained and accessible on this object:
    pipeline_cv = GridSearchCV(estimator=pipeline, param_grid=parameters, refit=True, cv=3, verbose=3,
                           scoring='f1_samples')
    
    return pipeline_cv


# function to print the classification report for the given label:
def display_results(y_test, y_test_pred, label):
    
    print("Classification report for category {}:\n".format(label))
    print(metrics.classification_report(y_test[label], y_test_pred[label]))

    print('')
    print('====================')
    print('')


def evaluate_model(model, X_test, Y_test, category_names):
    
    Y_test_pred = model.predict(X_test)
    Y_test_pred = pd.DataFrame(Y_test_pred, columns=Y_test.columns)
    for label in category_names:
        display_results(Y_test, Y_test_pred, label)


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet') # download for lemmatization
        
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
