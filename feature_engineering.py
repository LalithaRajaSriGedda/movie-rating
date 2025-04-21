import pandas as pd
def add_director_success_rate(df):
    df = df.copy()
    director_avg_rating = df.groupby('Director')['Rating'].mean().rename('director_success_rate')
    df = df.join(director_avg_rating, on='Director')
    return df

def add_avg_rating_of_similar_movies(df):
    df = df.copy()
    genre_avg = df.groupby('Genre')['Rating'].mean().rename('genre_avg_rating')
    df = df.join(genre_avg, on='Genre')
    return df

def engineer_features(df):
    df = add_director_success_rate(df)
    df = add_avg_rating_of_similar_movies(df)
    return df
