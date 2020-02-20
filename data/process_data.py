# Revised with flake8
import sys

import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath, sep=',', index_col='id')
    categories = pd.read_csv(categories_filepath, sep=',', index_col='id')

    df = messages.join(categories, on='id', how='inner')
    categories = df.categories.str.split(";", expand=True)

    category_colnames = [
        cat.split('-')[0]
        for cat in list(categories.loc[2, :])
        ]

    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split("-", expand=True)[1]

        # convert column from string to numeric
        categories[column] = categories[column].astype('int32')

    df = df.drop(['categories'], axis=1, inplace=False)
    df = df.join(categories, on='id', how='inner')

    # drop duplicates
    df = df.drop_duplicates(inplace=False)

    return df


def clean_data(df):
    # Generate dummy cols for categorical variable
    df = pd.concat([df, pd.get_dummies(df['genre'], dummy_na=False)], axis=1)
    df.drop(['genre'], axis=1, inplace=True)

    # Category "related" with value = 2 is equal to value = 0
    df.loc[df['related'] == 2, 'related'] = 0

    # Remove categories with only one value
    cols_one_value = []
    for col in df.columns[2:]:
        if (df[col] == 1).all() or (df[col] == 0).all():
            cols_one_value.append(col)

    print(
        "This columns has only one value and it will be discarded: {}"
        .format(cols_one_value)
        )

    df.drop(cols_one_value, axis=1, inplace=True)
    return df


def save_data(df, database_filename):
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql(
        'messages_processed',
        engine,
        index=False,
        if_exists='replace'
    )


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = \
            sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories \
              datasets as the first and second argument respectively, as \
              well as the filepath of the database to save the cleaned data \
              to as the third argument. \n\nExample: python process_data.py \
              disaster_messages.csv disaster_categories.csv \
              DisasterResponse.db')


if __name__ == '__main__':
    main()
