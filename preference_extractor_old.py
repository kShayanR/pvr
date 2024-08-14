import pandas as pd

users_df = pd.read_csv('datasets/users.csv')
reviews_df = pd.read_csv('datasets/reviews.csv')
details_df = pd.read_csv('datasets/details.csv')

user_reviews = reviews_df.groupby('user_id')['text'].apply(' '.join).reset_index()

from sklearn.feature_extraction.text import TfidfVectorizer

details_df['combined_text'] = details_df['amenities'].fillna('') + ' ' + details_df['description'].fillna('')
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
details_tfidf = vectorizer.fit_transform(details_df['combined_text'])

user_reviews_tfidf = vectorizer.transform(user_reviews['text'])

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

similarity_matrix = cosine_similarity(user_reviews_tfidf, details_tfidf)

top_similarities = np.argmax(similarity_matrix, axis=1)
user_reviews['top_hotel_id'] = details_df.iloc[top_similarities]['location_id'].values

user_reviews['preferences'] = details_df.iloc[top_similarities]['combined_text'].values

users_df = users_df.merge(user_reviews[['user_id', 'preferences']], on='user_id', how='left')

users_df.to_csv('datasets/users_preferences_old.csv', index=False)