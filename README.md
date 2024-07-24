# Movie Recommendation System

This is a simple Movie Recommendation System built using Python. The project recommends movies based on user preferences and is built using data processing and machine learning techniques.

## Introduction

The Movie Recommendation System uses a collaborative filtering approach to suggest movies to users based on their past interactions with the movie database. This system utilizes the cosine similarity metric to identify similarities between movies.

## Features

- Recommends movies based on cosine similarity.
- User-friendly interface built with Streamlit.
- Uses a precomputed similarity matrix for efficient recommendations.

## Requirements

- Python 3.6+
- Streamlit
- pandas
- numpy
- scikit-learn
- pickle

## Data

The dataset used in this project is from [TMDB Movie Metadata](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata-dataset). The data is preprocessed to create a similarity matrix, which is stored in the `similarity.pkl` file. The `movie_dict.pkl` file contains the movie dictionary used for lookup and display purposes.

