import json
from fastapi import FastAPI, Path
from fastapi.responses import JSONResponse
import json
import User_Based 
import MovieRecommendation_Model as recmodel
import ast
import uvicorn
import requests
import pandas as pd

app = FastAPI()

@app.get('/status')
def health_check():
    return 'Success'

@app.get('/popular')
def popular_movies():
    popular_mov = []
    df_mov = recmodel.popularMovies()
    # popular_mov = df_mov.index.values
    # df_mov.index.values
    df_dict = df_mov.to_dict('records')
    return JSONResponse(content=df_dict)

@app.get('/user_based')
def recommend_user_based(usr_id:int, num_rec:int):
    mov_list = []
    movId_list = []
    df = recmodel.user_based_recommend(user_id=usr_id, m=num_rec)
    data = df.to_dict()
    for i in data['title']:
        mov_list.append(data['title'][i])
    for j in data['movieId']:
            movId_list.append(data['movieId'][j])
    df_temp = pd.DataFrame({'movieId':movId_list, 'title':mov_list})
    df_dict = df_temp.to_dict('records')
    return JSONResponse(content=df_dict)

@app.get('/content_based')
def recommend_content(movie:str):
    return recmodel.contents_based_recommender(movie)

@app.get('/item_based')
def recommend_item(movie_name:str):
    data = recmodel.item_based_recommend(movie_name)
    data_dict = data.to_dict('records')
    return JSONResponse(content=data_dict)

@app.get('/images/{tmdbId}')
def grab_poster(tmdbId:int):
    response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=f45fba83edb23d739754015d36dfc0ab&language=en-US'.format(tmdbId))
    data = response.content
    data = data.decode('UTF-8')
    data = data.replace('false','False')
    data = data.replace('null','""')
    data = ast.literal_eval(data)
    return 'https://image.tmdb.org/t/p/w500{}'.format(data['poster_path'])
     
