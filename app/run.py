from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar
from joblib import load
from sqlalchemy import create_engine

import json
import plotly
import re
import pandas as pd
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

stop_words = stopwords.words("english")
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
app = Flask(__name__)


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


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages_processed', engine)

# load model
# model = joblib.load("../models/classifier.pkl")
model = load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """
    Description: This is the web index function.

    Arguments:
        None

    Returns:
        None

    """
    # extract data needed for visuals
    cols = list(df.columns)
    cols.remove('message')
    cols.remove('original')

    graphs = []
    for col in cols:
        graphs.append(
            {
                'data': [
                    Bar(
                        x=[0, 1],
                        y=df.groupby(col).count()['message']
                    )
                ],

                'layout': {
                    'title': f'Distribution of {col}',
                    'yaxis': {
                        'title': "Count"
                    },
                    'xaxis': {
                        'title': col
                    }
                }
            }
        )

    bestParams = model.best_params_

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template(
        'master.html',
        ids=ids,
        graphJSON=graphJSON,
        bestParams=bestParams
    )


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """
    Description: This is the function that handles user query
    and displays model results

    Arguments:
        None

    Returns:
        None

    """
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
    """
    Description: Main function. Starts the web application.

    Mandatory arguments:
        None

    Returns:
        None
    """
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
