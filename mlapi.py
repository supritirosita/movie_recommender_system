# -*- coding: utf-8 -*-
"""
Supriti Rosita
Reg. No.: 200968260
Batch 4
"""
#Importing libraries
import uvicorn
from fastapi import FastAPI
import mlmodels

#Creating app object
app = FastAPI()

#Index, route
@app.get('/')
def index():
    return {'message' : 'Recommender System - Supriti Rosita'}

#Movies from Similar Genres:
@app.get('/Genre Based Recommender')
def predict_movies(movie: str):
    df = mlmodels.get_recommendations_based_on_genres(movie)
    
    return df,{'message' : 'The top 10 recommendations based on similar genres are:'}

#Content Based Model:
@app.get('/Content Based Movie Recommender System')
def predict_content(user_id: int):
    df = mlmodels.get_recommendation_content_model(user_id)
    
    return df,{'message' : 'The Recommendations for the User are:'}

#Item-Item Based Model:
@app.get('/Item-Item CF Based Movie Recommender System')
def predict_item(user_id: int):
    df = mlmodels.recommendedMoviesAsperItemSimilarity(user_id)
    dfitem = mlmodels.movieIdToTitle(df)
    return dfitem,{'message' : 'The Top 7 Recommendations for the User are:'}

#User-Item Based Model:
@app.get('/User-Item CF Based Movie Recommender System')
def predict_user(user_id: int):
    df = mlmodels.getRecommendedMoviesAsperUserSimilarity(user_id)
    dfuser = mlmodels.movieIdToTitle(df)
    return dfuser,{'message' : 'The Top 7 Recommendations for the User are:'}

#Matrix Factorization:
@app.get('/SVD Based Movie Recommender System')
def predict_svd(user_id: int):
    df = mlmodels.recommend_movies(user_id)
    
    return df,{'message' : 'The Top 20 Recommendations for the User are:'}
    
#Hybrid Model:
#@app.get('/Hybrid Movie Recommender System')
#def predict_from_userid(user_id: int):
    #df=mlfinal.hybrid_content_svd_model(user_id)
    #return df,{'message': 'The top 10 recommendations for the User are:'}
    
# Run the api with uvicorn
if __name__ == '__mlapi__':
    uvicorn.run(app, host = '127.0.0.1', port = 8000)
    
#uvicorn fast1: app --reload
