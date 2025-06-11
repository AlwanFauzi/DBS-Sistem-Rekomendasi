import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping

import os
import random

pd.set_option('display.max_columns', None)

# --- Load Dataset ---
movies_df = pd.read_csv('data/movies.csv')
ratings_df = pd.read_csv('data/ratings.csv')

# --- EDA (ringkas, hanya print info penting) ---
print("Info movies:")
print(movies_df.info())
print("\nInfo ratings:")
print(ratings_df.info())
print("\nNull values in movies:\n", movies_df.isnull().sum())
print("\nNull values in ratings:\n", ratings_df.isnull().sum())
print("\nDescriptive stats ratings:\n", ratings_df.describe())

# --- Genre Analysis ---
movies_df['genre_list'] = movies_df['genres'].apply(lambda x: x.split('|') if isinstance(x, str) else [])
from collections import Counter
genre_counter = Counter([genre for genres in movies_df['genre_list'] for genre in genres])
genre_df = pd.DataFrame(genre_counter.items(), columns=['Genre', 'Jumlah']).sort_values(by='Jumlah', ascending=False)
print("\nTop genres:\n", genre_df.head())

# --- Data Preparation ---
# Content-Based Filtering Preparation
vectorizer = CountVectorizer(token_pattern=r'[^|]+')
genre_matrix = vectorizer.fit_transform(movies_df['genres'])
cosine_sim = cosine_similarity(genre_matrix)
indices = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()

# Collaborative Filtering Preparation
ratings = ratings_df.drop('timestamp', axis=1)
user_enc = LabelEncoder()
movie_enc = LabelEncoder()
ratings['user'] = user_enc.fit_transform(ratings['userId'].values)
ratings['movie'] = movie_enc.fit_transform(ratings['movieId'].values)
num_users = ratings['user'].nunique()
num_movies = ratings['movie'].nunique()
print(f"\nJumlah unik user: {num_users}")
print(f"Jumlah unik movie: {num_movies}")

# --- Content-Based Filtering Function ---
def recommend_movies_cbf(title, num_recommendations=10):
    if title not in indices:
        print("Judul tidak ditemukan.")
        return pd.DataFrame()
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies_df.iloc[movie_indices][['title', 'genres']]

# --- Precision@K untuk evaluasi CBF ---
def precision_at_k(recommended_titles, reference_title, k=5):
    ref_genres = set(movies_df[movies_df['title'] == reference_title]['genres'].iloc[0].split('|'))
    relevant = 0
    for title in recommended_titles[:k]:
        rec_genres = set(movies_df[movies_df['title'] == title]['genres'].iloc[0].split('|'))
        if len(ref_genres & rec_genres) > 0:
            relevant += 1
    return relevant / k

# --- Recall@K untuk evaluasi CBF ---
def recall_at_k(recommended_titles, reference_title, k=5):
    ref_genres = set(movies_df[movies_df['title'] == reference_title]['genres'].iloc[0].split('|'))
    total_relevant = len(ref_genres)
    found_genres = set()
    for title in recommended_titles[:k]:
        rec_genres = set(movies_df[movies_df['title'] == title]['genres'].iloc[0].split('|'))
        found_genres.update(ref_genres & rec_genres)
    return len(found_genres) / total_relevant if total_relevant > 0 else 0

# Contoh penggunaan CBF dan evaluasi
print("\nRekomendasi CBF untuk 'Toy Story (1995)':")
cbf_result = recommend_movies_cbf('Toy Story (1995)', 10)
print(cbf_result)

recommended_titles = cbf_result['title'].tolist()
precision5 = precision_at_k(recommended_titles, 'Toy Story (1995)', k=5)
recall5 = recall_at_k(recommended_titles, 'Toy Story (1995)', k=5)
print(f"\nPrecision@5: {precision5:.2f}")
print(f"Recall@5: {recall5:.2f}")

# --- Collaborative Filtering (RecommenderNet) ---
X = ratings[['user', 'movie']].values
y = ratings['rating'].values
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

user_input = Input(shape=(1,))
movie_input = Input(shape=(1,))
user_embedding = Embedding(num_users, 50)(user_input)
movie_embedding = Embedding(num_movies, 50)(movie_input)
user_vec = Flatten()(user_embedding)
movie_vec = Flatten()(movie_embedding)
dot = Dot(axes=1)([user_vec, movie_vec])
output = Dense(1, activation='linear')(dot)
model = Model([user_input, movie_input], output)
model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

print("\nTraining Collaborative Filtering model...")
history = model.fit([x_train[:,0], x_train[:,1]], y_train,
                    validation_data=([x_test[:,0], x_test[:,1]], y_test),
                    epochs=10, batch_size=64, verbose=1)

# --- Evaluasi Collaborative Filtering ---
y_pred = model.predict([x_test[:,0], x_test[:,1]]).flatten()
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\nRMSE Collaborative Filtering: {rmse:.4f}")
print("""
Interpretasi:
1. RMSE mengukur seberapa jauh rata-rata prediksi model terhadap nilai rating sebenarnya.
2. Semakin rendah RMSE → semakin baik prediksi model.
3. RMSE ≈ 1.0 tergolong cukup baik, mengingat skala rating adalah 0.5 - 5.0.
""")

# --- Personalized Recommendation Function ---
def recommend_movies_cf(user_id, num_recommendations=10):
    if user_id not in ratings_df['userId'].values:
        print("User ID tidak ditemukan.")
        return pd.DataFrame()
    encoded_user_id = user_enc.transform([user_id])[0]
    movie_ids = np.arange(num_movies)
    user_array = np.full(shape=num_movies, fill_value=encoded_user_id)
    predictions = model.predict([user_array, movie_ids]).flatten()
    top_indices = predictions.argsort()[-num_recommendations:][::-1]
    recommended_movie_ids = movie_enc.inverse_transform(top_indices)
    return movies_df[movies_df['movieId'].isin(recommended_movie_ids)][['movieId', 'title', 'genres']]

# Contoh penggunaan CF
print("\nRekomendasi CF untuk user_id=100:")
print(recommend_movies_cf(user_id=100, num_recommendations=10))