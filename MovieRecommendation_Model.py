# importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from fuzzywuzzy import fuzz
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# importing the dataset
movies_df = pd.read_csv('movies.csv')
tags_df = pd.read_csv('tags.csv')
ratings_df = pd.read_csv('ratings.csv')
links_df = pd.read_csv('links.csv')

# most popular movies with highest avg rating
ratings_movies_df = pd.merge(movies_df, ratings_df, on='movieId')
def popularMovies():
    X = ratings_movies_df.groupby('title').rating.count()
    Y = ratings_movies_df.groupby('title').rating.mean()
    rating_title = pd.DataFrame(data=X)
    rating_title['ratingAvg'] = pd.DataFrame(Y)
    rating_title.sort_values('rating', ascending=False)
    filtering_df = rating_title[rating_title['rating']>50]
    filtering_df.sort_values('ratingAvg', ascending=False, inplace=True)
    # filtering_df.drop(['rating','ratingAvg'], axis=1, inplace=True)
    filtering_df['title'] = filtering_df.index
    filtering_df = filtering_df.set_index('rating')
    return filtering_df.head(20)

# merging all the datasets
movieLens = pd.merge(left=movies_df, right=ratings_df, on='movieId')

# creating the pivot table
user_ratings_item = ratings_df.pivot_table(index='movieId',columns='userId',values='rating')
# user_ratings_item

# dropping users who have rated less than 50 movies
user_ratings_item = user_ratings_item.dropna(thresh=50, axis=1).fillna(0)
# user_ratings_item

# creating a csr matrix to reduce the computations
csr_data = csr_matrix(user_ratings_item.values)
user_ratings_item.reset_index(inplace=True)

# using cosine similarity method in KNN
model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10, n_jobs=-1)
model.fit(csr_data)

# recommender function
def item_based_recommend(title):
    n_movie_to_recommend = 20
    movie_list = movies_df[movies_df['title'].str.contains(title, case=False)]  
    if len(movie_list):        
        movie_idx = movie_list.iloc[0]['movieId']
        movie_idx = user_ratings_item[user_ratings_item['movieId'] == movie_idx].index[0]
        distances, indices = model.kneighbors(csr_data[movie_idx], n_neighbors=n_movie_to_recommend+1)    
        rec_movies = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
        recommend_frame = []
        for val in rec_movies:
            movie_idx = user_ratings_item.iloc[val[0]]['movieId']
            idx = movies_df[movies_df['movieId'] == movie_idx].index
            recommend_frame.append({'Title':movies_df.iloc[idx]['title'].values[0],'Distance':val[1],'movieId':int(movie_idx)})
        df = pd.DataFrame(recommend_frame, index=range(1, n_movie_to_recommend+1))
        df.sort_values(by=['Distance'], inplace=True)
        return df.reset_index(drop=True)
    else:
        return "No similar movies found :("


# movie_input = input("Enter a movie you liked: ")

# print("Top 10 movies similar to", str(movie_input), "and that were liked by other users: ")
# item_based_recommend(movie_input)

# the function to extract titles 
def extract_title(title): 
   year = title[len(title)-5:len(title)-1]
   
   # some movies do not have the info about year in the column title. So, we should take care of the case as well.
   
   if year.isnumeric():
      title_no_year = title[:len(title)-7]
      return title_no_year
   else:
      return title
      
# the function to extract years
def extract_year(title):
   year = title[len(title)-5:len(title)-1]
   # some movies do not have the info about year in the column title. So, we should take care of the case as well.
   if year.isnumeric():
      return int(year)
   else:
      return np.nan
# change the column name from title to title_year
movies_df.rename(columns={'title':'title_year'}, inplace=True) 

# remove leading and ending whitespaces in title_year
movies_df['title_year'] = movies_df['title_year'].apply(lambda x: x.strip()) 

# create the columns for title and year
movies_df['title'] = movies_df['title_year'].apply(extract_title) 
movies_df['year'] = movies_df['title_year'].apply(extract_year) 

# removing the unnecessary characters in the 'genres' column
movies_df['genres'] = movies_df['genres'].str.replace('|',' ')
movies_df['genres'] = movies_df['genres'].str.replace('Sci-Fi','SciFi')
movies_df['genres'] = movies_df['genres'].str.replace('Film-Noir','Noir')

# intialising a TfidfVectorizer object with stop_word as 'English' as our data was taken in English
tfidf_vector = TfidfVectorizer(stop_words='english')

# apply the object to the 'genres' column
tfidf_matrix = tfidf_vector.fit_transform(movies_df['genres'])

# printing the vectorized 'genres' column
# print(list(enumerate(tfidf_vector.get_feature_names())))

# create the cosine similarity matrix
sim_matrix = linear_kernel(tfidf_matrix,tfidf_matrix) 
# print(sim_matrix)

# function to find the closest title
def matching_score(a,b):
   return fuzz.ratio(a,b)

# a function to convert index to title_year
def get_title_year_from_index(index):
   return movies_df[movies_df.index == index]['title_year'].values[0]

# a function to convert index to title
def get_title_from_index(index):
   return movies_df[movies_df.index == index]['title'].values[0]

# a function to convert title to index
def get_index_from_title(title):
   return movies_df[movies_df.title == title].index.values[0]
   
# a function to return the most similar title to the words a user type
def find_closest_title(title):
   leven_scores = list(enumerate(movies_df['title'].apply(matching_score, b=title)))
   sorted_leven_scores = sorted(leven_scores, key=lambda x: x[1], reverse=True)
   closest_title = get_title_from_index(sorted_leven_scores[0][0])
   distance_score = sorted_leven_scores[0][1]
   return closest_title, distance_score

def contents_based_recommender(movie_user_likes):
   return_mov = []
   closest_title, distance_score = find_closest_title(movie_user_likes)
   # When a user does not make misspellings
   if distance_score == 100:
      movie_index = get_index_from_title(closest_title)
      movie_list = list(enumerate(sim_matrix[int(movie_index)]))
      # remove the typed movie itself
      similar_movies = list(filter(lambda x:x[0] != int(movie_index), sorted(movie_list,key=lambda x:x[1], reverse=True))) 
      
      # print('Here\'s the list of movies similar to '+'\033[1m'+str(closest_title)+'\033[0m'+'.\n')
      # for i,s in similar_movies[:how_many]:
         # print(get_title_year_from_index(i))
   # When a user makes misspellings    
   else:
      # print('Did you mean '+'\033[1m'+str(closest_title)+'\033[0m'+'?','\n')
      movie_index = get_index_from_title(closest_title)
      movie_list = list(enumerate(sim_matrix[int(movie_index)]))
      similar_movies = list(filter(lambda x:x[0] != int(movie_index), sorted(movie_list,key=lambda x:x[1], reverse=True)))
      # print('Here\'s the list of movies similar to: '+'\033[1m'+str(closest_title)+'\033[0m'+'.\n')
      for i,s in similar_movies[:20]:
         return_mov.append(get_title_year_from_index(i))
   return return_mov

# contents_based_recommender('Toy Story', 10)

# merging the datasets
user_rating_user = pd.merge(movies_df, ratings_df, on='movieId').drop('timestamp', axis=1)
# user_rating_user

# making a pivot table
user_pivot_table = user_rating_user.pivot_table(index='userId', columns='movieId', values='rating')
# user_pivot_table

# normalizing the ratings
user_pivot_norm = user_pivot_table.subtract(user_pivot_table.mean(axis=1), axis = 'rows')
# user_pivot_norm.head()

# using pearson correlation to get similar users
user_sim_corr = user_pivot_norm.T.corr()
# user_sim_corr

def get_similar_user(pick_user_id,n):
    user_similarity_threshold = 0.3
    similar_users = user_sim_corr[user_sim_corr[pick_user_id]>user_similarity_threshold][pick_user_id].sort_values(ascending=False)[:n]
    return similar_users

def user_based_recommend(user_id,m=20):

    # getting the top n similar users
    n=10
    sim_users = get_similar_user(user_id,n)

    # remove movies that have been watched
    picked_userid_watched = user_pivot_norm[user_pivot_norm.index == user_id].dropna(axis=1, how='all')

    # remove movies that none of the similar users have watched
    similar_user_movies = user_pivot_norm[user_pivot_norm.index.isin(sim_users.index)].dropna(axis=1, how='all')

    # remove the already watched movies by the user from the movie list
    similar_user_movies.drop(picked_userid_watched.columns,axis=1, inplace=True, errors='ignore')

    # dictionary to store item scores
    item_score = {}
    
    # loop through items
    for i in similar_user_movies.columns:
        # get the ratings for movie i
        movie_rating = similar_user_movies[i]
        # variable to store the score
        total = 0
        # variable to store the number of scores
        count = 0
        # loop through similar users
        for j in sim_users.index:
            # if the movie has rating
            if pd.isna(movie_rating[j]) == False:
                # score is the sum of user similarity score multiply by the movie rating
                score = sim_users[j] * movie_rating[j]
                # add the score to the total score for the movie so far
                total += score
                # add 1 to the count
                count +=1
        # get the average score for the item
        item_score[i] = total / count
    # convert dictionary to pandas dataframe
    item_score = pd.DataFrame(item_score.items(), columns=['movieId', 'movie_score'])

    # sort the movies by score
    ranked_item_score = item_score.sort_values(by='movie_score', ascending=False)
    ranked_item_score = pd.merge(ranked_item_score, movies_df, on='movieId').drop(['genres','title_year','year'],axis=1)
    return ranked_item_score.head(m)

# recommend = user_based_recommend(29,10)
# recommend





