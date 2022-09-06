#!/usr/bin/env python
# coding: utf-8

# # Supriti Rosita - 200968260 - ML Project Phase 2

# # Movie Recommendation System
# Recommender systems are one of the most successful and widespread application of machine learning technologies in business.
# One can find large scale recommender systems in retail, video on demand, or music streaming.

# ## Contents
# - Content based filtering
# - Collaborative Filtering
#     - Memory based collaborative filtering
#         - User-Item Filtering
#         - Item-Item Filtering
#     - Model based collaborative filtering
#         - Single Value Decomposition(SVD)
# - Evaluating Collaborative Filtering using SVD
# - Hybrid Model

# ## Importing Libraries

# In[1]:


from math import sqrt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# ## Dataset: Movielens(ml-latest-small: ratings,movies)

# In[2]:


# Reading ratings file
ratings = pd.read_csv("ratings.csv")

# Reading movies file
movies = pd.read_csv("movies.csv")


# In[3]:


df_movies = movies 
df_ratings = ratings 


# # Content based filtering
# The concepts of Term Frequency (TF) and Inverse Document Frequency (IDF) are used in content based filtering mechanisms and also in information retrieval systems(such as a content based recommender). They are used to determine the relative importance of a document / article / news item / movie etc.
# 
# ### Term Frequency (TF) and Inverse Document Frequency (IDF)
# TF: the frequency of a word in a document. 
# IDF: the inverse of the document frequency among the whole corpus of documents. 
# 

# For calculating distances, widely used similarity coefficients such as Euclidean, Cosine, Pearson Correlation etc., are calculated.
# 
# In this case cosine distance is used. Here the point of interest in similarity-higher the value more similar they are. (Since the below function gives us the distance, it will be deducted from 1.) 
# 
# **Cosine similarity**
# is a measure of similarity between two non-zero vectors of an inner product space that measures the cosine of the angle between them.
# 

# In[4]:


# Define a TF-IDF Vectorizer Object.
tfidf_movies_genres = TfidfVectorizer(token_pattern = '[a-zA-Z0-9\-]+')

#Replace NaN with an empty string
df_movies['genres'] = df_movies['genres'].replace(to_replace="(no genres listed)", value="")

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_movies_genres_matrix = tfidf_movies_genres.fit_transform(df_movies['genres'])

cosine_sim_movies = linear_kernel(tfidf_movies_genres_matrix, tfidf_movies_genres_matrix)
#print(cosine_sim_movies)


# In[5]:


def get_recommendations_based_on_genres(movie_title, cosine_sim_movies=cosine_sim_movies):
    """
    Calculates top 10 movies to recommend based on given movie titles genres. 
    :param movie_title: title of movie to be taken for base of recommendation
    :param cosine_sim_movies: cosine similarity between movies 
    :return: Titles of movies recommended to user
    """
    # Get the index of the movie that matches the title
    idx_movie = df_movies.loc[df_movies['title'].isin([movie_title])]
    idx_movie = idx_movie.index
    
    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores_movies = list(enumerate(cosine_sim_movies[idx_movie][0]))
    
    # Sort the movies based on the similarity scores
    sim_scores_movies = sorted(sim_scores_movies, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores_movies = sim_scores_movies[1:10]
    
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores_movies]
    
    # Return the top 10 most similar movies
    return df_movies['title'].iloc[movie_indices]


# In[6]:


#get_recommendations_based_on_genres("Father of the Bride Part II (1995)")


# In[7]:


def get_recommendation_content_model(userId):
    """
    Calculates top movies to be recommended to user based on movie user has watched.  
    :param userId: userid of user
    :return: Titles of movies recommended to user
    """
    recommended_movie_list = []
    movie_list = []
    df_rating_filtered = df_ratings[df_ratings["userId"]== userId]
    for key, row in df_rating_filtered.iterrows():
        movie_list.append((df_movies["title"][row["movieId"]==df_movies["movieId"]]).values) 
    for index, movie in enumerate(movie_list):
        for key, movie_recommended in get_recommendations_based_on_genres(movie[0]).iteritems():
            recommended_movie_list.append(movie_recommended)

    # removing already watched movie from recommended list    
    for movie_title in recommended_movie_list:
        if movie_title in movie_list:
            recommended_movie_list.remove(movie_title)
    
    return set(recommended_movie_list)


# In[8]:


# rc=input("enter user id:")
# rcm=int(rc)
# get_recommendation_content_model(rcm)


# In[9]:


from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation


# In[10]:


df_movies = movies 
df_ratings = ratings 


# ## Collaborative Filtering
# Types of collaborative filtering techniques
# * Memory based 
#  - User-Item Filtering
#  - Item-Item Filtering
# * Model based 
#  - Matrix Factorization

# ### Memory Based Approach
# 

# #### Item-Item Filtering
# 

# #### Implementation (Item-Item Filtering)

# In[11]:


df_movies_ratings=pd.merge(df_movies, df_ratings)


# In[12]:


df_movies_ratings.head()


# In[13]:


df_movies_ratings.shape


# Pivot table function is used for one to one maping between movies, user and their rating. 
# (The default pivot_table command takes average if we have multiple values of one combination.)

# In[50]:


ratings_matrix_items = df_movies_ratings.pivot_table(index=['movieId'],columns=['userId'],values='rating').reset_index(drop=True)
ratings_matrix_items.fillna( 0, inplace = True )
ratings_matrix_items.values


# In[15]:


# ratings_matrix_items.head()


# In[16]:


# ratings_matrix_items.shape


# In[17]:


movie_similarity = 1 - pairwise_distances( ratings_matrix_items.values, metric="cosine" )
#Filling diagonals with 0s for future use when sorting is done
np.fill_diagonal( movie_similarity, 0) 
ratings_matrix_items = pd.DataFrame( movie_similarity )
ratings_matrix_items


# The function below stake the movie name as a input and finds the movies which are similar to this movie.
# This function first find the index of movie in movies frame and then take the similarity of movie and align in movies dataframe; therefore it helps determine the similarity of the movie with all other movies.

# In[18]:


def item_similarity(movieName): 
    """
    recomendates similar movies
   :param data: name of the movie 
   """
    try:
        #user_inp=input('Enter the reference movie title based on which recommendations are to be made: ')
        user_inp=movieName
        inp=df_movies[df_movies['title']==user_inp].index.tolist()
        inp=inp[0]

        df_movies['similarity'] = ratings_matrix_items.iloc[inp]
        df_movies.columns = ['movie_id', 'title', 'release_date','similarity']
    except:
        print("Sorry, the movie is not in the database!")


# Here, user id of the user for whom movies should be recommended is provided:
# First, the movies which are rated 5 or 4.5 by the user for whom we want to recommend movies is found. (since, in Item-Item similarity, movies are recommended to the user based on their previous selection.)
# 
# Thus, to foster our algorithm movies that are most-liked/highly rated by the user are found and on its basis similar movies are recommended.
# 
# The function has appended the similarity of the movie highly rated by the user to the movies data frame.
# The frame is sorted in descending order to list out the movies which are highly similar to movie highly rated by the user.
# Upon filtering the movies that are most similar, if similarity is greater than 0.45 then the movie is considered.
# One final filtering process is done to reccomend to the user, the movies he has not seen.

# In[19]:


def recommendedMoviesAsperItemSimilarity(user_id):
    """
     Recommending movie which user hasn't watched as per Item Similarity
    :param user_id: user_id to whom movie needs to be recommended
    :return: movieIds to user 
    """
    user_movie= df_movies_ratings[(df_movies_ratings.userId==user_id) & df_movies_ratings.rating.isin([5,4.5])][['title']]
    user_movie=user_movie.iloc[0,0]
    item_similarity(user_movie)
    sorted_movies_as_per_userChoice=df_movies.sort_values( ["similarity"], ascending = False )
    sorted_movies_as_per_userChoice=sorted_movies_as_per_userChoice[sorted_movies_as_per_userChoice['similarity'] >=0.45]['movie_id']
    recommended_movies=list()
    df_recommended_item=pd.DataFrame()
    user2Movies= df_ratings[df_ratings['userId']== user_id]['movieId']
    for movieId in sorted_movies_as_per_userChoice:
            if movieId not in user2Movies:
                df_new= df_ratings[(df_ratings.movieId==movieId)]
                df_recommended_item=pd.concat([df_recommended_item,df_new])
            best7=df_recommended_item.sort_values(["rating"], ascending = False )[0:8] 
    return best7["movieId"]


# In[20]:


def movieIdToTitle(listMovieIDs):
    """
     Converting movieId to titles
    :param user_id: List of movies
    :return: movie titles
    """
    movie_titles= list()
    for id in listMovieIDs:
        movie_titles.append(df_movies[df_movies['movie_id']==id]['title'])
    return movie_titles


# In[48]:


# user_id=50
# print("Recommended movies,:\n",movieIdToTitle(recommendedMoviesAsperItemSimilarity(user_id)))


# #### User-Item Filtering
# 

# #### Implementation (User-Item Filtering)

# A similar matrix to that of Item-Item Similarity will be created but here, rows will be userId and columns will be movieId-since we want to detemine a vector of different users.
# In similar ways distance and similarity between users is found.

# In[22]:


ratings_matrix_users = df_movies_ratings.pivot_table(index=['userId'],columns=['movieId'],values='rating').reset_index(drop=True)
ratings_matrix_users.fillna( 0, inplace = True )
movie_similarity = 1 - pairwise_distances( ratings_matrix_users.values, metric="cosine" )
#Filling diagonals with 0s for future use when sorting is done
np.fill_diagonal( movie_similarity, 0 ) 
ratings_matrix_users = pd.DataFrame( movie_similarity )
ratings_matrix_users


# In the above matrix, similarity of users in columns with respective users in row. So the user with highest similarity can be found by finding maximum value in a column. Thus, a pair of similar users can be determined.

# In[23]:


# ratings_matrix_users.idxmax(axis=1)


# In[24]:


# ratings_matrix_users.idxmax(axis=1).sample( 10, random_state = 10 )


# In[25]:


similar_user_series= ratings_matrix_users.idxmax(axis=1)
df_similar_user= similar_user_series.to_frame()


# In[26]:


# df_similar_user.columns=['similarUser']


# In[27]:


# df_similar_user


# The function below takes the Id of the user to whom movie reccomendations should be made. It is done by finding similar users and filtering movies which are highly rated by those users and recommending them to the given user.

# In[51]:


movieId_recommended=list()
def getRecommendedMoviesAsperUserSimilarity(userId):
    """
     Recommending movies which user hasn't watched as per User Similarity
    :param user_id: user_id to whom movie needs to be recommended
    :return: movieIds to user 
    """
    user2Movies= df_ratings[df_ratings['userId']== userId]['movieId']
    sim_user=df_similar_user.iloc[0,0]
    df_recommended=pd.DataFrame(columns=['movieId','title','genres','userId','rating','timestamp'])
    for movieId in df_ratings[df_ratings['userId']== sim_user]['movieId']:
        if movieId not in user2Movies:
            df_new= df_movies_ratings[(df_movies_ratings.userId==sim_user) & (df_movies_ratings.movieId==movieId)]
            df_recommended=pd.concat([df_recommended,df_new])
        best7=df_recommended.sort_values(['rating'], ascending = False )[0:8]  
    return best7['movieId']


# In[29]:


# user_id=50
# recommend_movies= movieIdToTitle(getRecommendedMoviesAsperUserSimilarity(user_id))
# print("Movies you should watch are:\n")
# print(recommend_movies)


# ### Single Value Decomposition

# Singular value decomposition is a method of decomposing a matrix into three other matrices:
# <img src="https://cdn-images-1.medium.com/max/1600/0*i4rDKIAE0o1ZXtBd.">

# Where:
# 
#     A is an m × n matrix
#     U is an m × r orthogonal matrix
#     S is an r × r diagonal matrix
#     V is an r × n orthogonal matrix

# ### Matrix Factorization
# 

# ### Implementation

# In[30]:


# Import libraries
import numpy as np
import pandas as pd

# Reading ratings file
ratings = pd.read_csv("ratings.csv")

# Reading movies file
movies = pd.read_csv("movies.csv")


# In[31]:


n_users = ratings.userId.unique().shape[0]
n_movies = ratings.movieId.unique().shape[0]
print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_movies))


# To format of our ratings matrix to be one row per user and one column per movie we pivot ratings and call the new variable Ratings (with a capital *R).

# In[32]:


Ratings = ratings.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)
Ratings.head()


# In[33]:


#de-normalizing the data (normalize by each users mean) and converting it from a dataframe to a numpy array.
R = Ratings.values
print(R)
user_ratings_mean = np.mean(R, axis = 1)
print(user_ratings_mean.size)
Ratings_demeaned = R - user_ratings_mean.reshape(-1, 1) 
## Making the user_ratings_mean vertical by reshaping


# ### Model-Based Collaborative Filtering

# ### SVD (Setup)
# Scipy and Numpy both have functions to do the singular value decomposition. In this case the Scipy function svds is used because it let’s us choose how many latent factors we want to use to approximate the original ratings matrix (instead of having to truncate it after).

# In[34]:


from scipy.sparse.linalg import svds


# In[35]:


U, sigma, Vt = svds(Ratings_demeaned, k = 50)


# In[36]:


print('Size of sigma: ' , sigma.size)


# Since we are leveraging matrix multiplication to get predictions, we willl convert the $\Sigma$ (now are values) to the diagonal matrix form.

# In[37]:


sigma = np.diag(sigma)


# In[38]:


# print('Shape of sigma: ', sigma.shape)
# print(sigma)


# In[39]:


# print('Shape of U: ', U.shape)
# print('Shape of Vt: ', Vt.shape)


# ### Making Predictions from the Decomposed Matrices
# 
# We now have everything we need to make movie ratings predictions for every user. This is done by following the math and matrix multiply $U$, $\Sigma$, and $V^{T}$ back to get the rank $k=50$ approximation of $A$.
# 
# But first, the user means needs to be added back to get the actual star ratings prediction.

# In[40]:


all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)


# In[41]:


print('All user predicted rating : ', all_user_predicted_ratings.shape)


# With the predictions matrix for every user, we can build a function to recommend movies for any user. We return the list of movies the user has already rated, for the sake of comparison.
# 
# We will use the column names from the ratings df

# In[42]:


print('Rating Dataframe column names', Ratings.columns)


# In[43]:


preds = pd.DataFrame(all_user_predicted_ratings, columns = Ratings.columns)
preds.head()


# A function to return the movies with the highest predicted rating that the specified user hasn't already rated needs to be developed. Though any explicit movie content features (such as genre or title) wasn't used, we merge in that information to get a more complete picture of the recommendations.

# In[44]:


def recommend_movies(predictions, userID, movies, original_ratings, num_recommendations):
    """
    Implementation of SVD by hand
    :param predictions : The SVD reconstructed matrix, 
    userID : UserId for which you want to predict the top rated movies, 
    movies : Matrix with movie data, original_ratings : Original Rating matrix, 
    num_recommendations : num of recos to be returned
    :return: num_recommendations top movies
    """ 
    # Get and sort the user's predictions
    user_row_number = userID - 1 # User ID starts at 1, not 0
    sorted_user_predictions = predictions.iloc[user_row_number].sort_values(ascending=False) # User ID starts at 1
    
    # Get the user's data and merge in the movie information.
    user_data = original_ratings[original_ratings.userId == (userID)]
    user_full = (user_data.merge(movies, how = 'left', left_on = 'movieId', right_on = 'movieId').
                     sort_values(['rating'], ascending=False)
                 )

    print('User {0} has already rated {1} movies.'.format(userID, user_full.shape[0]))
    print('Recommending highest {0} predicted ratings movies not already rated.'.format(num_recommendations))
    
    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    recommendations = (movies[~movies['movieId'].isin(user_full['movieId'])].
         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
               left_on = 'movieId',
               right_on = 'movieId').
         rename(columns = {user_row_number: 'Predictions'}).
         sort_values('Predictions', ascending = False).
                       iloc[:num_recommendations, :-1]
                      )

    return user_full, recommendations


# Recommedning 20 movies for user with ID 150

# In[45]:


#already_rated, predictions = recommend_movies(preds, 150, movies, ratings, 20)


# In[46]:


# Top 20 movies that User 1310 has rated 
#already_rated.head(20)


# In[47]:


# Top 20 movies that User 1310 hopefully will enjoy
#predictions


# These look like pretty good recommendations. Although the genre of the movie wasn't used as a feature, the truncated matrix factorization features "picked up" on the underlying tastes and preferences of the user. The model has recommended some Action, Adventure, Romance, Thriller movies - all of which were genres of some of this user's top rated movies.

# In[ ]:




