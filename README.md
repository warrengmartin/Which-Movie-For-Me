# Which-Movie-For-Me

This repository contains a Python implementation of a collaborative filtering model for movie recommendations, built using PyTorch. This model is based on Jeremy Howards model on Kaggle: https://www.kaggle.com/code/jhoward/collaborative-filtering-deep-dive

## Description

This code implements a collaborative filtering model using PyTorch to provide movie recommendations based on user ratings. The model learns from user-item interactions in the MovieLens dataset to predict preferences and suggest relevant movies to new users.

## Dataset

The code utilizes the MovieLens dataset, specifically the "ml-latest-small" version, which is automatically downloaded within the script. This dataset contains user ratings for movies.

## Model

The core of the system is a neural collaborative filtering model implemented as a PyTorch Module. The model learns embeddings for both users and items (movies) and predicts ratings based on the interaction of these embeddings.

## Key Features

* **Data Preprocessing:**
    * Downloads and extracts the MovieLens dataset.
    * Merges rating and movie data.
    * Creates a user-item interaction matrix.
    * Handles missing values using KNN imputation.
    * Normalizes ratings using min-max scaling.
* **Model Training:**
    * Defines a PyTorch Dataset and DataLoader for efficient data handling.
    * Implements a neural collaborative filtering model with user and item embeddings.
    * Uses Mean Squared Error (MSE) loss and the Adam optimizer for training.
    * Includes gradient clipping to prevent exploding gradients.
* **Prediction and Recommendation:**
    * Loads the trained model.
    * Takes new user ratings as input.
    * Predicts ratings for all movies for the new user.
    * Recommends the top-rated movies based on the predictions.
* **Model Saving and Loading:**
    * Saves the trained model to a file for later use.
    * Provides a function to load the saved model and make predictions.

## How to Use

1. Clone the repository: `git clone https://github.com/your-username/collaborative-filtering-movie-recommendation.git`
2. Navigate to the project directory: `cd collaborative-filtering-movie-recommendation`
3. Install the required libraries: `pip install -r requirements.txt`
4. Run the Python script: `python collaborative_filtering_recommendation.py`

The code will download the dataset, train the model, save it, and then make predictions for a new user with randomly generated ratings. You can modify the `new_user_ratings` DataFrame in the script to provide your own ratings and test the recommendations.

## Future Improvements

* Deploy the model as a web application for interactive recommendations. 
