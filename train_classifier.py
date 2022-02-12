import sys
import pandas as pd
import sqlite3
import pickle
from sqlalchemy import create_engine
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

def load_data(database_filepath):
    '''
    Description: Loads data from the SQLite database
    
    Input: Database name
    
    Output: Features(X), labels(y) and cateogry_names
    '''
    conn = sqlite3.connect('DisasterResponse.db')
    df = pd.read_sql('SELECT * FROM DisasterResponsedata', conn)
    X = df['message'] 
    y = df[['related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military','water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']]
    category_names = y.columns.tolist()
    return X, y, category_names

def tokenize(text):
    '''
    Description: Normalise, lemmatize and tokenize text from messages.
    
    Input: Text data
    
    Output: Normalised, lemmatized and tokenized text
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model():
    '''
    Description: Create text processing and machine learning pipeline that uses the custom tokenize function in the machine learning pipeline to vectorize and transform text. MultiOutputClassifier to support multi-target classification using LinearSVC classifier to enables predictions on 36 categories.
        
    Output: Text processing and machine learning pipeline 
    
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(LinearSVC()))])
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Description: Use ML pipeline to predict labels of test features and produce classification report containing precision, recall, f1 score for each category.
    
    Input: ML pipeline, test and label features and category_names
    
    Output: F1 score, precision and recall for each category in test set 
    
    '''
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred,target_names=category_names))

def save_model(model, model_filepath):
    '''
    Description: Exports the final model as a pickle file
    
    Input: ML pipeline, name of pickle file
    
    Output: Pickle file
    '''    
    filename = model_filepath
    
    return pickle.dump(model, open(filename, 'wb'))


def main():
    
    if len(sys.argv) == 3:
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