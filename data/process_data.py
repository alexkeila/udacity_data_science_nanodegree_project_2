# import libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Function to load the datasets, merge them together and save it on a dataframe
    
    Parameters:
    messages_filepath (str): path to the messages dataset
    categories_filepath (str): path to the categories dataset

    Returns:
    dataframe: result of the loaded and merged data
    """
    
    # load messages dataset:
    df_messages = pd.read_csv(messages_filepath)
    
    # load categories dataset:
    df_categories = pd.read_csv(categories_filepath)
    
    # merge datasets:
    df = pd.merge(left=df_messages, right=df_categories, on='id', how='inner')
    
    return df


def clean_data(df):
    """
    Function to clean the dataset and save it on a new dataframe
    
    Parameters:
    df (dataframe): data to be cleaned

    Returns:
    dataframe: result of the cleaned data
    """
    
    # create a dataframe of all the individual category columns:
    categories = df['categories'].str.split(pat=';', expand=True)
    
    # select the first row of the categories dataframe:
    row = categories.head(1)

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [name[:-2] for name in list(row.to_dict(orient='row')[0].values())]
    
    # rename the columns of `categories`:
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string:
        categories[column] = categories[column].astype(str).str[-1]

        # convert column from string to numeric:
        categories[column] = categories[column].astype(int)
        
    # drop the original categories column from `df`:
    df.drop('categories', axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # remove duplicates:
    df.drop_duplicates(inplace=True)
    
    # the 'related' column contains 3 unique values (0, 1, 2)
    # convert to a binary category by forcing the observations with related = 2 to be = 1,
    # since it is a multilabel classification, but not a multi-output classification (each category is binary):
    df['related'] = np.where(df['related'] == 2, 1, df['related'])
    
    return df


def save_data(df, database_filename):
    """
    Function to save a dataset on the chosen database
    
    Parameters:
    df (dataframe): data to be saved on the database
    database_filename (str): path to where the database is located on disk

    Returns:
    None
    """

    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('DisasterMessages', engine, if_exists='replace', index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
