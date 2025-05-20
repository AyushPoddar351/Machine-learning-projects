#!/usr/bin/env python3
# train_random_forest.py

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def main():
    # Load dataset
    df = pd.read_csv('bengaluru_land_prices.csv')

    # Drop irrelevant column
    if 'Pin Code' in df.columns:
        df.drop(columns=['Pin Code'], inplace=True)

    # Handle missing values
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Separate features and target
    target = 'Total Price (INR)'
    X = df.drop(columns=[target])
    y = df[target]

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Build preprocessing pipeline
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

    # Build complete pipeline with Random Forest
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))
    ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Train model
    model_pipeline.fit(X_train, y_train)

    # Save the trained model to a pickle file
    with open('random_forest_model.pkl', 'wb') as f:
        pickle.dump(model_pipeline, f)

    print("Training complete. Model saved to 'random_forest_model.pkl'.")

if __name__ == '__main__':
    main()
