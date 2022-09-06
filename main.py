# -*- coding: utf-8 -*-
"""
Akash Yadav
Reg. No.: 200968184
Batch 4
"""

# Importing libraries
import uvicorn
from fastapi import FastAPI
import userBasedRecommender

# Creating app object
app = FastAPI()

# Index, route
@app.get('/')
def index():
    return {'message' : 'Recommender System - Akash Yadav'}

@app.get('/User Based Movie Prediction')
def predict_movies_from_user(user_id: int):
    df = userBasedRecommender.userBased_Recommender(user_id)
        
    return df
    
    
# Run the api with uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host = '127.0.0.1', port = 8000)
    
#uvicorn main: app --reload