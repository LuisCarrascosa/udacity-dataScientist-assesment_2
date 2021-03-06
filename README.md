# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Summary:
#### Libraries version:
With "version.py" it's possible to generate a file "versions.txt" that contains all libraries versions used.
"versions_submitted,txt" has the versions used to test the project.

#### ETL pipeline:
We use two datasets with tweets messages categorized:
* disaster_messages.csv
* disaster_categories.csv

We merge both datasets and create columns for all categories. Drop duplicates and clean data:

1. Generate dummy cols for categorical variable "genre"
2. Category "related" with value = 2 is equal to value = 0
3. Remove categories with only one value

Then we save the data with SQLAlchemy.

#### ML pipeline:
1. Building model
2. Training model. Uses GridSearchCV to hyperparameters search of CountVectorizer and TfidfTransformer. Uses a RandomForestClassifier as classificator. GridSearchCV use recall_score on recall because the data is very unbalanced
3. Saving model
4. Evaluating model