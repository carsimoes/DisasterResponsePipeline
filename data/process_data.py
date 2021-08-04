import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    #df = messages.merge(categories, how='inner', on='id')
    df = pd.merge(categories, messages, on='id')
    return df


def clean_data(df):
    # Split `categories` into separate category columns.
    categories = df['categories'].str.split(';', expand=True)

    # Cut the last character of each category
    # select the first row of the categories dataframe
    row = categories.head(1)
    category_colnames = row.applymap(lambda x: x[:-2]).iloc[0, :]
    category_colnames = category_colnames.tolist()
    #categories.columns = row.apply(lambda x:x[:-2])

    # Rename the columns of `categories`
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1.
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, join='inner')

    # Drop the duplicates.
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('MessagesFromDisasters', engine, index=False)


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