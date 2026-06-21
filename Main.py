# File: Main.py
# Author: Natalia González Ramos
# Created On: 01/02/2026
# Purpose:

import pandas as pd
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer


def main():
    # Load datasets from CSV files
    train_data = pd.read_csv('real-estate-train.csv', header=0)
    test_data = pd.read_csv('real-estate-test.csv', header=0)

    # Indicate what data is the train data and what is the test data
    feature_columns = ['X1', 'X2', 'X3', 'X4']
    target_column = 'Y'

    X_train = train_data[feature_columns]
    Y_train = train_data[target_column]

    X_test = test_data[feature_columns]
    Y_test = test_data[target_column]


    # Set up preprocessing strategies
    preprocessors = {
        "None": None,
        "StandardScaler" : StandardScaler(),
        "Normalizer L1": Normalizer(norm="l1"),
        "Normalizer L2": Normalizer(norm="l2")
    }
    # Set up regression models
    models = {
        "LinearRegression": LinearRegression(),
        "SGDRegressor": SGDRegressor()
    }

    # Loop that matches each preprocessor with each model once
    for prep_name, preprocessor in preprocessors.items():
        for model_name, model in models.items():

            if preprocessor is None:
                pipeline = Pipeline([
                    ("model", model)
                ])
            else:
                pipeline = Pipeline([
                    ("preprocess", preprocessor),
                    ("model", model)
                ])

            # Fit the pipeline
            pipeline.fit(X_train, Y_train)

            train_score = pipeline.score(X_train, Y_train)
            test_score = pipeline.score(X_test, Y_test)

            # Print solutions
            print(f"Preprocessing: {prep_name} | Model: {model_name} " +
                  f"| Train Score: {train_score:.4f} | Test Score: {test_score:.4f}")

if __name__ == '__main__':
    main()
