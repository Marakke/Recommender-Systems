import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import random

# Load MovieLens ratingsset
ratings = pd.read_csv('ml-latest-small/ratings.csv')

# Group of 3 users
user_group = [1, 2, 3]

# Filter ratings for the selected user group
group_ratings = ratings[ratings['userId'].isin(user_group)]

# Pivot the ratings to create a user-item matrix
ratings_matrix = group_ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Instantiate and fit a kNN model for collaborative filtering
k = min(10, len(user_group) - 1)  # Number of neighbors for kNN
knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(ratings_matrix)

# Function to generate recommendations for the user group in 3 sequences
def generate_group_recommendations(user_group, ratings_matrix, knn_model, k, top_n=10, num_sequences=3):
    group_recommendations = []
    for sequence in range(num_sequences):
        sequence_recommendations = []
        for user_id in user_group:
            user_idx = ratings_matrix.index.get_loc(user_id)
            distances, indices = knn_model.kneighbors(ratings_matrix.iloc[user_idx, :].values.reshape(1, -1),
                                                      n_neighbors=k + 1)
            random_indices = random.sample(range(1, len(indices.squeeze())), k)  # Randomly select k neighbors
            similar_users_indices = indices.squeeze()[random_indices]
            similar_users_ratings = ratings_matrix.iloc[similar_users_indices]
            user_movies = ratings_matrix.iloc[user_idx]
            unrated_movies = user_movies[user_movies == 0].index  # Filter unrated movies
            avg_ratings = similar_users_ratings.mean(axis=0)
            avg_ratings = avg_ratings[unrated_movies]  # Consider only unrated movies
            sequence_recommendations.extend(avg_ratings.sort_values(ascending=False).index[:top_n].tolist())
        group_recommendations.append(sequence_recommendations)
    return group_recommendations

# Generate recommendations for the user group in 3 sequences
group_top_movies = generate_group_recommendations(user_group, ratings_matrix, knn_model, k, num_sequences=3)

# Display top-10 recommendations for the user group in 3 sequences
for sequence, movies_sequence in enumerate(group_top_movies, start=1):
    print(f"Sequence {sequence} Top 10 movies for the user group to watch together:")
    print(movies_sequence[:10])
