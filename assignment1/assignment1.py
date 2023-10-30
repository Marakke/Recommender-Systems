import pandas as pd
from scipy.stats import pearsonr

# Compute Pearson correlation between users
def pearson_similarity(user1, user2):
    common_movies = user_item_matrix.loc[user1].mul(user_item_matrix.loc[user2], fill_value=0)
    if len(common_movies) == 0:
        return 0
    corr, _ = pearsonr(user_item_matrix.loc[user1], user_item_matrix.loc[user2])
    return corr

# Find similar users
def find_similar_users(target_user, num_users=5):
    similarities = []
    for user in user_item_matrix.index:
        if user != target_user:
            similarity = pearson_similarity(target_user, user)
            similarities.append((user, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:num_users]

# Movie recommendations for a user
def recommend_movies(user):
    similar_users = find_similar_users(user)
    recommendations = {}
    
    for similar_user, similarity in similar_users:
        movies_rated_by_similar_user = user_item_matrix.loc[similar_user]
        for movie, rating in movies_rated_by_similar_user.items():
            if rating > 0 and user_item_matrix.loc[user, movie] == 0:
                if movie in recommendations:
                    recommendations[movie].append((similar_user, similarity))
                else:
                    recommendations[movie] = [(similar_user, similarity)]
    
    recommended_movies = []
    for movie, similar_users in recommendations.items():
        total_similarity = sum(similarity for _, similarity in similar_users)
        if total_similarity != 0:  # Added check for zero similarity
            weighted_rating = sum(similarity * user_item_matrix.loc[similar_user_id, movie] for similar_user_id, similarity in similar_users)
            predicted_rating = weighted_rating / total_similarity
            recommended_movies.append((movie, predicted_rating))
    
    recommended_movies.sort(key=lambda x: x[1], reverse=True)
    return recommended_movies

# Get similar users and recommended movies for a user
def get_recommendations_for_user(user, num_similar_users=10, num_recommended_movies=10):
    similar_users = find_similar_users(user, num_similar_users)
    
    similar_users_list = [user_id for user_id, _ in similar_users]
    
    recommended_movies = recommend_movies(user)
    
    recommended_movies_list = [movie_id for movie_id, _ in recommended_movies[:num_recommended_movies]]
    
    return similar_users_list, recommended_movies_list

print("Assignment 1 part (a)")

# Ratings dataset
ratings = pd.read_csv('ml-latest-small/ratings.csv')

print(ratings.head())
print(f"Number of ratings: {len(ratings)}")

user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

print("Assignment 1 part (b)")

# Test user IDs
user1 = 1
user2 = 2

similarity = pearson_similarity(user1, user2)
print(f"Pearson correlation between User {user1} and User {user2}: {similarity}")

print("Assignment 1 part (c)")

recommended_movies = recommend_movies(user1)
print(f"Recommended movies for user {user1}:")
for movie_id, predicted_rating in recommended_movies[:10]:
    print(f"Movie ID: {movie_id}, Predicted Rating: {predicted_rating}")

print("Assignment 1 part (d)")

similar_users_list, recommended_movies_list = get_recommendations_for_user(user1)
print(f"Similar users for user {user1}: {similar_users_list}")
print(f"Recommended movies for user {user1}: {recommended_movies_list}")


print("Assignment 1 part (e)")
