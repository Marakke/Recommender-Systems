import pandas as pd
from sklearn.neighbors import NearestNeighbors

"""
Purpose:
This function generates group movie recommendations in multiple sequences. It leverages a k-Nearest Neighbors (kNN) approach 
to find movies similar to those rated by the users in the group. The function provides both the recommendations and 
additional information useful for explaining the recommendation process.

How it Works:
1. Iterates through a specified number of sequences (num_sequences).
2. For each sequence, it fits a kNN model to the ratings_matrix, which contains user ratings for various movies.
3. For each user in the user group, the function identifies similar users based on their movie ratings and then
   finds the top N unrated movies based on these similar users' average ratings.
4. The function keeps track of both the movies considered and the selected top N movies for each sequence.

Reasoning:
The use of multiple sequences with a kNN model for each allows capturing varied aspects of user preferences. By considering
unrated movies from similar users, the recommendations are likely to align with the users' interests while maintaining
diversity. Tracking both considered and selected movies enables detailed explanations for the recommendation logic.
"""
def generate_group_recommendations_with_info(user_group, ratings_matrix, top_n=10, num_sequences=3):
    group_recommendations = []
    recommendation_info = {'considered_movies': {}, 'selected_movies': {}}

    for sequence in range(num_sequences): 
        knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
        knn_model.fit(ratings_matrix)

        sequence_recommendations = []
        sequence_considered_movies = {}  # Track considered movies for each sequence

        for user_id in user_group: 
            user_idx = ratings_matrix.index.get_loc(user_id)
            # Use all other users as potential neighbors
            distances, indices = knn_model.kneighbors(ratings_matrix.iloc[user_idx, :].values.reshape(1, -1), n_neighbors=len(ratings_matrix) - 1)
            
            # Flatten indices and distances, exclude the first one (self)
            flat_indices = indices.flatten()[1:]
            flat_distances = distances.flatten()[1:]

            # Pair distances with indices and sort them
            sorted_neighbors = sorted(zip(flat_distances, flat_indices))

            # Select top N similar users based on sorted distances
            similar_users_indices = [idx for _, idx in sorted_neighbors[:top_n]]

            similar_users_ratings = ratings_matrix.iloc[similar_users_indices]
            user_movies = ratings_matrix.iloc[user_idx]
            unrated_movies = user_movies[user_movies.isna() | (user_movies == 0)].index
            avg_ratings = similar_users_ratings.mean(axis=0)
            avg_unrated_ratings = avg_ratings[unrated_movies]

            # Update considered movies with average ratings
            sequence_considered_movies.update(avg_unrated_ratings.to_dict())

            # Select top N movies based on average ratings, not randomly
            top_movies = avg_unrated_ratings.nlargest(top_n).index.tolist()
            sequence_recommendations.extend(top_movies)

            # Update selected movies
            for movie in top_movies:
                recommendation_info['selected_movies'].setdefault(movie, []).append(avg_unrated_ratings[movie])

        group_recommendations.append(sequence_recommendations)
        recommendation_info['considered_movies'][sequence] = sequence_considered_movies

    return group_recommendations, recommendation_info


"""
Purpose:
Provides an explanation for why a specific movie (atomic case) was or was not recommended to the user group. This function
is crucial for understanding the reasons behind the inclusion or exclusion of individual movies in the recommendation list.

How it Works:
1. Checks if the movie was selected for recommendation. If so, confirms its recommendation.
2. If the movie was not selected, the function explains why, based on its average rating compared to the group's average
   rating for that movie or its alignment with the group's overall preferences.
3. If the movie was not even considered, it informs the user accordingly.

Reasoning:
This function enhances the transparency of the recommendation process by providing specific reasons for the presence or
absence of a movie. Understanding why a certain movie was not recommended despite being popular or highly rated can be
insightful for users, especially in a group setting with diverse tastes.
"""
def explain_atomic_case(movie_id, recommendation_info, ratings_matrix, movies_data):
    # Handle the case where the movie ID is not in the movies_data dictionary
    title = movies_data[movie_id]['title'] if movie_id in movies_data else f"Movie ID {movie_id}"
    
    if movie_id in recommendation_info['selected_movies']:
        return f"'{title}' was recommended."
    elif movie_id in recommendation_info['considered_movies']:
        # Ensure avg_rating is retrieved correctly
        avg_ratings_list = recommendation_info['considered_movies'].get(movie_id, [])
        avg_rating = sum(avg_ratings_list) / len(avg_ratings_list) if avg_ratings_list else 0

        # Calculate the group's average rating for the movie
        group_avg_rating = ratings_matrix.get(movie_id, pd.Series()).mean()

        # Determine the reason for not selecting the movie
        reason = "lower than group's average rating" if avg_rating < group_avg_rating else "not aligning with group's preferences"
        return f"'{title}' was considered but not selected due to {reason}."
    else:
        return f"'{title}' was not considered in the recommendation process."



"""
Purpose:
Explains why movies of a particular genre were or were not recommended to the user group. It addresses group cases in the
recommendation process, shedding light on genre-based preferences and decisions.

How it Works:
1. The function iterates through the movies considered and selected for recommendation, filtering them based on the specified
   genre.
2. It then compiles lists of movie titles from the selected and considered categories that match the genre.
3. Based on these lists, it provides an explanation about the inclusion or exclusion of movies from the specified genre.

Reasoning:
Group preferences often include genre preferences. This function clarifies how well the recommended movies align with the
genre interests of the group. By identifying specific movies within the genre that were considered or selected, it offers
a clear and detailed insight into the recommendation logic, particularly as it pertains to genre preferences.
"""
def explain_group_case(genre, movies_data, recommendation_info):
    considered_titles = []
    for movie_id in recommendation_info['considered_movies']:
        if movie_id in movies_data and genre in movies_data[movie_id]['genres']:
            considered_titles.append(movies_data[movie_id]['title'])

    selected_titles = []
    for movie_id in recommendation_info['selected_movies']:
        if movie_id in movies_data and genre in movies_data[movie_id]['genres']:
            selected_titles.append(movies_data[movie_id]['title'])

    if selected_titles:
        return f"Movies from the genre '{genre}' like {', '.join(selected_titles)} were recommended."
    elif considered_titles:
        return f"Movies from the genre '{genre}' like {', '.join(considered_titles)} were considered but not selected."
    else:
        return f"No movies from the genre '{genre}' were considered."



"""
Purpose:
This function explains why a specific movie was not ranked first in the recommendation list. It addresses the position 
absenteeism case, providing insights into the ranking logic within the group recommendation process.

How it Works:
1. Checks if the movie was recommended but not ranked first.
2. If so, it explains the reason, which could be due to diversity considerations, the presence of other movies with higher
   average ratings, or a better alignment with the group's overall preferences.
3. If the movie was not recommended at all, it defaults to the atomic case explanation for comprehensive coverage.

Reasoning:
The ranking of movies in a recommendation list is as important as the selection itself, especially in a group setting where
preferences might vary. This function adds depth to the explanation system by not just confirming whether a movie was 
recommended, but also delving into the reasons behind its specific ranking, thereby offering a more nuanced understanding
of the recommendation process.
"""
def explain_position_absenteeism(movie_id, recommendation_info, ratings_matrix, movies_data):
    title = movies_data[movie_id]['title'] if movie_id in movies_data else f"Movie ID {movie_id}"
    
    if movie_id in recommendation_info['selected_movies']:
        # Ensure avg_rating is a single scalar value
        avg_ratings_list = recommendation_info['selected_movies'].get(movie_id, [])
        avg_rating = sum(avg_ratings_list) / len(avg_ratings_list) if avg_ratings_list else 0

        # Calculate the group's average rating for the movie, ensuring it is a single scalar value
        group_avg_rating = ratings_matrix.get(movie_id, pd.Series()).mean()

        # Determine the reason for not ranking the movie first
        reason = ("diversity considerations" if avg_rating < group_avg_rating 
                  else "there were movies with higher average ratings or better matching the group's preferences")
        return f"'{title}' was recommended but not ranked first due to {reason}."
    else:
        return explain_atomic_case(movie_id, recommendation_info, ratings_matrix, movies_data)




# Datasets
links = pd.read_csv('ml-latest-small/links.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')
tags = pd.read_csv('ml-latest-small/tags.csv')


# Test user group
user_group = [1, 2, 3]

# Filter ratings for the selected user group
group_ratings = ratings[ratings['userId'].isin(user_group)]

# Pivot the ratings to create a user-item matrix
ratings_matrix = group_ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Generate recommendations for the user group in 3 sequences with diversified aggregation
group_top_movies, additional_info = generate_group_recommendations_with_info(user_group, ratings_matrix, num_sequences=4)

movies_data = {}  # This should be a dictionary with movie IDs as keys and their details (including genre) as values

for index, row in movies.iterrows():
    movie_id = row['movieId']
    movies_data[movie_id] = {
        'title': row['title'],
        'genres': row['genres'].split('|')
    }

print('Assignment 4')

for sequence, movies_sequence in enumerate(group_top_movies, start=1):
    print(f"Sequence {sequence} Top 10 movies for the user group to watch together:")
    print(movies_sequence[:10])
    for movie_id in movies_sequence[:10]:
        print(f"Movie title: {movies_data[movie_id]['title']}")

print("1. Explain the atomic case for the movie 'Matrix':")
print(explain_atomic_case(2571, additional_info, ratings_matrix, movies_data))
print("1. Explain the atomic case for the movie 'Toy Story':")
print(explain_atomic_case(1, additional_info, ratings_matrix, movies_data))
print("1. Explain the atomic case for the movie 'The Godfather':")
print(explain_atomic_case(858, additional_info, ratings_matrix, movies_data))
print("1. Explain the atomic case for the movie 'Death Race 2000':")
print(explain_atomic_case(7991, additional_info, ratings_matrix, movies_data))

print("2. Explain the group case for the genre 'Action':") 
print(explain_group_case('Action', movies_data, additional_info))
print("2. Explain the group case for the genre 'Comedy':")
print(explain_group_case('Comedy', movies_data, additional_info))
print("2. Explain the group case for the genre 'Horror':")
print(explain_group_case('Horror', movies_data, additional_info))

print("3. Explain the position absenteeism for the movie 'Matrix':")
print(explain_position_absenteeism(2571, additional_info, ratings_matrix, movies_data))
print("3. Explain the position absenteeism for the movie 'Toy Story':")
print(explain_position_absenteeism(1, additional_info, ratings_matrix, movies_data))
print("3. Explain the position absenteeism for the movie 'The Godfather':")
print(explain_position_absenteeism(858, additional_info, ratings_matrix, movies_data))
print("3. Explain the position absenteeism for the movie 'Death Race 2000':")
print(explain_position_absenteeism(7991, additional_info, ratings_matrix, movies_data))

