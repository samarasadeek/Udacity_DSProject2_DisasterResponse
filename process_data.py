import sys
import pandas as pd 
from sqlalchemy import create_engine
import sqlite3

def load_data(messages_filepath, categories_filepath):
    '''
    Description: Loads 2 datasets from csv files and then merges 2 datasets into 1 dataframe.
    
    Input: 2 csv files
    
    Output: 1 dataframe
    '''
    messages = pd.read_csv(messages_filepath) 
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='left', on='id')
    return df

def clean_data(df):
    '''
    Description
    : Each element in the 'categories' column contains the list of 36 possible categories, and a '- 1' or '- 0' that indicates the category of the message (in the 'messages' column); the categories are separated by ';'. 
    : A new dataframe called categories is created that contains a column for each category
    : Each column entry is a 1 or 0, which was extracted from 'categories' column of the input dataframe. 
    : The 'categories' column from the input dataframe is dropped.
    : The 'categories' dataframe and the input dataframe are concatenated.
    : Duplicate rows and 'child_alone' column dropped. 'child_alone' column dropped since no messages in the dataset was tagged under this category. 
    
    Input: dataframe 
    
    Output: modified dataframe 
    '''    
    categories = df['categories'].str.split(';', expand=True)
    row = categories.loc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0]).values.tolist()
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).apply(lambda x:x.split('-')[1])
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # drop the original categories column from `df`
    df.drop('categories',axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)
    
    # drop duplicates
    df.drop_duplicates(subset=['id'], inplace=True)
    
    # replace entry 2 with 1 in 'related' column
    df['related'].replace(2,1,inplace=True)
    
    # drop child_alone column
    df.drop('child_alone', axis=1, inplace=True)
    
    return df

def save_data(df, database_path):
    '''
    Description: Stores data in table in a a specified SQLite database.
    
    Input: Dataframe contained wrangled data 
         : Name of database    
    
    Ouput: Database with data table

    '''    
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql('DisasterResponsedata', engine, index=False, if_exists='replace')

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