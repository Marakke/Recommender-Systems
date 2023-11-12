import pandas as pd
from scipy.stats import pearsonr

ratings = pd.read_csv('ml-latest-small/ratings.csv')

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
        if total_similarity != 0:
            weighted_rating = sum(similarity * user_item_matrix.loc[similar_user_id, movie] for similar_user_id, similarity in similar_users)
            predicted_rating = weighted_rating / total_similarity
            recommended_movies.append((movie, predicted_rating))
    
    recommended_movies.sort(key=lambda x: x[1], reverse=True)
    return pd.DataFrame(recommended_movies, columns=['MovieID', 'Predicted Rating'])

user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

def average_aggregation(group_recommendations):
    return group_recommendations.mean(axis=0)

def least_misery_aggregation(group_recommendations):
    return group_recommendations.min(axis=0)

def generate_group_recommendations(user_ids, aggregation_method):
    group_recommendations = pd.DataFrame()
    
    for user_id in user_ids:
        # Compute individual recommendations using user-based collaborative filtering
        individual_recommendations = recommend_movies(user_id)
        group_recommendations[user_id] = individual_recommendations['Predicted Rating']

    # Aggregate recommendations using the specified method
    group_aggregated = aggregation_method(group_recommendations)
    
    # Display top 10 movies with the highest predicted scores based on the aggregation method
    top_recommendations = group_aggregated.sort_values(ascending=False).head(10)
    return top_recommendations

def calculate_disagreement_score(user1, user2):
    common_movies = user_item_matrix.loc[user1].mul(user_item_matrix.loc[user2], fill_value=0)
    disagreement = (user_item_matrix.loc[user1] - user_item_matrix.loc[user2]).abs()
    disagreement = disagreement[common_movies != 0]
    return disagreement.mean()

def modify_recommendations_based_on_disagreements(group_recommendations, user_ids):
    total_disagreement = 0
    count = 0
    for i in range(len(user_ids)):
        for j in range(i+1, len(user_ids)):
            disagreement_score = calculate_disagreement_score(user_ids[i], user_ids[j])
            if pd.notna(disagreement_score):
                total_disagreement += disagreement_score
                count += 1

    if count > 0:
        average_disagreement = total_disagreement / count
        group_recommendations *= (1 - average_disagreement)

    return group_recommendations

def generate_group_recommendations_with_disagreement(user_ids, aggregation_method):
    group_recommendations = pd.DataFrame()
    
    for user_id in user_ids:
        individual_recommendations = recommend_movies(user_id)
        group_recommendations[user_id] = individual_recommendations['Predicted Rating']

    group_recommendations = modify_recommendations_based_on_disagreements(group_recommendations, user_ids)
    group_aggregated = aggregation_method(group_recommendations)
    
    top_recommendations = group_aggregated.sort_values(ascending=False).head(10)
    return top_recommendations

print("Assignment 2 part (a)")

# Test user group
group_of_users = [1, 2, 3]

# Average Method
average_recommendations = generate_group_recommendations(group_of_users, average_aggregation)
print("Top 10 Recommendations (Average Method):")
print(average_recommendations)

# Least Misery Method
misery_recommendations = generate_group_recommendations(group_of_users, least_misery_aggregation)
print("\nTop 10 Recommendations (Least Misery Method):")
print(misery_recommendations)

print("Assignment 2 part (b)")

# Modified Recommendations with Disagreement
disagreement_recommendations = generate_group_recommendations_with_disagreement(group_of_users, average_aggregation)
print("\nTop 10 Recommendations Considering Disagreements:")
print(disagreement_recommendations)
