from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, recall_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from joblib import dump, load
from sklearn.model_selection import train_test_split, GridSearchCV

import sys
import re
import numpy as np
import pandas as pd
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

stop_words = stopwords.words("english")
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def load_data(database_filepath):
    """
    Description: This function loads the data.

    Arguments:
        database_filename: path of database file.

    Returns:
        Dataframe: Indepent variable
        Dataframe: Dependents variables
        list: Dependents variables columns names
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("messages_processed", con=engine)

    df = pd.read_sql_table("messages_processed", con=engine)

    X = df['message']
    Y = df.drop(['message', 'original'], axis=1, inplace=False)

    return X, Y, list(Y.columns)


def tokenize(text):
    """
    Description: This function process text. Replaces urls, applies lower, 
    tokenize, lemmatizer and removes stop words.

    Arguments:
        text: text to tokenize.

    Returns:
        clean_tokens: Indepent variable

    """
    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)

    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')

    # tokenize text
    text = text.lower()
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens if word not in stop_words
        ]

    return clean_tokens


def build_model():
    """
    Description: This function builds the model. Uses GridSearchCV to
    hyperparameters search of CountVectorizer and TfidfTransformer.
    Uses a RandomForestClassifier as classificator.
    GridSearchCV use recall_score on recall because the data is very unbalanced

    Arguments:
        None.

    Returns:
        Model

    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

#     parameters = {
#         'vect__ngram_range': ((1, 1), (1, 2)),
#         'vect__max_df': (0.5, 0.75, 1.0),
#         'vect__max_features': (None, 5000, 10000),
#         'tfidf__use_idf': (True, False),
#         'clf__estimator__n_estimators': [50, 100, 200],
#         'clf__estimator__min_samples_split': [2, 3, 4],
#         'clf__estimator__class_weight': ['balanced']
#     }

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 1.0),
        'vect__max_features': (None, 5000),
        'tfidf__use_idf': (True, False),
        #   'clf__estimator__n_estimators': [200],
        # 'clf__estimator__min_samples_split': [4],
        'clf__estimator__class_weight': ['balanced']
    }

    cv = GridSearchCV(
        estimator=pipeline,
        scoring=make_scorer(recall_score, average='micro'),
        param_grid=parameters
    )

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Description: This function evaluates the model and prints the result

    Arguments:
        model: model to evaluate
        X_test: X-values to test
        Y_test: Y-values to test
        category_names: Category names

    Returns:
        None

    """
    y_pred = model.predict(X_test)

    for col in category_names:
        i_related = category_names.index(col)

        print(f"Column: {col}")
        print(classification_report(
            np.array(Y_test[col]),
            y_pred[:, i_related]
        ))


def save_model(model, model_filepath):
    """
    Description: This function saves the model

    Arguments:
        model: model to evaluate
        X_test: X-values to test
        Y_test: Y-values to test
        category_names: Category names

    Returns:
        None

    """
    dump(model, model_filepath)


def load_model(model_filepath):
    """
    Description: This function load the model

    Arguments:
        model_filepath: path of model file.

    Returns:
        Model

    """
    return load(model_filepath)


def main():
    """
    Description: Main function. Build, train, save and evaluate the model.

    Mandatory arguments:
        model_filepath: path of model file.
        database_filename: path of database file.

    Returns:
        None
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)

        X_train, X_test, Y_train, Y_test = train_test_split(
            X,
            Y,
            test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Best params...')
        print(model.best_params_)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database \
              as the first argument and the filepath of the pickle file to \
              save the model to as the second argument. \n\nExample: python \
              train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
