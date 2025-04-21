Movie Rating prediction
  
Table of Contents

[Project Overview](#Project Overview)

Obectives

Data Understanding
Exploratory Data Analysis
Modelling
Recommandations



Project Overview
The Movie Rating Prediction project involves analyzing a dataset containing information about Indian movies. The dataset includes details like movie name, year, duration, genre, rating, votes, director, and three main actors. The data will be used to build a predictive model for movie ratings and extract valuable insights from the movie industry.

Objectives
Develop a predictive model: Create a machine learning model to predict movie ratings based on the provided dataset.This is essentially a regression problem, where we aim to estimate the numerical movie ratings based on various features.
Identify influential factors: Analyze the dataset to determine which factors (e.g., genre, director, actors) have the most significant impact on movie ratings.
Provide actionable insights: Offer insights to the film industry stakeholders to make informed decisions about movie production, casting, and marketing.

Data Understanding
The dataset is obtained from Kaggle: IMDb India Movies

The dataset has the following columns:

Name: Movie name
Year: release year of the movies
Duration: Movie duration
Genre: Movie genre
Rating: Movie rating
Votes: Number of votes received
Director: Movie director
Actor 1: First main actor
Actor 2: Second main actor
Actor 3: Third main actor

Exploratory Data Analysis
EDA inlude Rating analysis, Top 10 Directors with the Most Movies Directed, Top 10 Actors with the Most Movie Appearances, Top 10 Directors with the Highest-Rated Movies, Top 10 Highly Rated Movie Genres, Top 10 Years with Highest Average Ratings and Duration vs Rating

Modeling
Baseline Model - Linear Regression
Second Model: Random Forest Model
Third Model: Gradient Boosting Regressor
Hyperparameter Tuning of Random Forest Model
The tuned Random Forest performed better than the tuned model in terms of model generalization and avoiding overfitting, due to its more balanced performance between training and test data. This will be the final model used for prediction.

Recommendations
Year: Stay current with trends for successful movie releases.
Votes: Encourage audience reviews for more reliable ratings.
Duration: Align with audience preferences for movie length.
Collaboration: Partner with established industry figures.
Genres: Invest in highly-rated genres like History and Romance.
Duration vs. Ratings: No strong duration-rating correlation.
Critical Acclaim: Aim for positive reviews and acclaim

Conclusion
In conclusion, this project provides valuable insights and a predictive model for movie rating prediction. The film industry can benefit from these findings to make data-driven decisions regarding movie production, casting, and marketing. The most influential factors identified are the year of release, the number of votes, and movie duration.








