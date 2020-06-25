import sys
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import string
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

import pickle

def load_data(database_filepath):
    '''
    INPUT: database_filepath - database path 
    OUTPUT: Load dataset from database
            Define feature and target variables X, y and category_names     
    '''
    
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message'].values
    y = df.iloc[:,4:].values
    category_names = df.iloc[:,4:].columns
    
    return X, y, category_names

def tokenize(text):
    '''
    INPUT: text - text data to be processed
    OUTPUT: clean_tokens - a list with the following process:
           1. normalize
           2. lemmatize
           3. stopwords and punctuation
           4. strip
    '''
    text = text.lower()
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        if tok not in stopwords.words('english') and tok not in list(string.punctuation):
            clean_tok = lemmatizer.lemmatize(tok, pos='v').strip()
            clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    '''
    INPUT: take in the message column as input
    OUTPUT: cv - classification results on the other 36 categories in the dataset
    '''
    
    pipeline = Pipeline([
            ('features', FeatureUnion([
                ('nlp_pipeline', Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer())
                ]))
            ])),
            ('mclf', MultiOutputClassifier(RandomForestClassifier(n_jobs=-1)))
        ])

    # specify parameters for grid search
    parameters = {
        'features__nlp_pipeline__tfidf__use_idf': (True, False),
        'mclf__estimator__n_estimators': [10, 100]
    }
    
    # create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, cv=2)
    
    return cv

def evaluate_model(model, X_test, y_test, category_names):
    '''
    INPUT: model - final model after GridSearchCV
           X_test and y_test - train split datasets 
           category_names - names of columns of the dataset df
    OUTPUT: Report the f1 score, precision and recall for each output category of the dataset
    '''
    
    y_pred = model.predict(X_test)
    for i, label in enumerate(category_names):
        print("Classification Report for {}:".format(label))
        print(classification_report(y_test[:,i], y_pred[:,i]))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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