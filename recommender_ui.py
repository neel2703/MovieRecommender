import streamlit as st
import ast
import requests
import pandas as pd
import pickle
import json

movies = pd.read_csv('movies.csv')
links = pd.read_csv('links.csv')

def home():
    st.title('Popular Movies')
    # st.write('Some of the most popular movies')
    # if st.button('Recommend'):
    response = requests.get('http://127.0.0.1:8000/popular')
    data = ast.literal_eval(response.content.decode('UTF-8'))
    data = pd.DataFrame(data)
    data = pd.merge(data,movies,on='title')
    data = pd.merge(data,links,on='movieId')
    display_movs(data)


def search():
    st.title('Search')
    movie_list = pickle.load(open('movie_list.pkl','rb'))
    input = st.text_input('Search for a movie')
    option = st.radio('Recommendation Type: ',('Item Based','Content Based'))
    if st.button('Recommend'):
        if option == 'Content Based':
            response = requests.get('http://127.0.0.1:8000/content_based?movie={}'.format(str(input)))
            data = ast.literal_eval(response.content.decode('UTF-8'))
            data = pd.DataFrame(data)
            data.rename(columns={0:'title'},inplace=True)
            # data['title'] = data.index
            # # data['movieId'] = data['movieId'].astype('int64')
            data = pd.merge(data,movies,on='title')
            data = pd.merge(data,links, on='movieId')
            data.drop(['movieId','imdbId'],inplace=True,axis=1)
            display_movs(data)
        elif option == 'Item Based':
            response = requests.get('http://127.0.0.1:8000/item_based?movie_name={}'.format(str(input)))
            # data = ast.literal_eval(response.content.decode('UTF-8'))
            data = pd.DataFrame(response.json())
            data = pd.DataFrame(data)
            data.rename(columns={'Title':'title'},inplace=True)
            data = pd.merge(data,movies,on='movieId')
            data = pd.merge(data,links,on='movieId')
            data.drop('title_x',axis=1,inplace=True)
            data.rename(columns={'title_y':'title'},inplace=True)
            display_movs(data)
            # st.table(data)
        else:
            pass
    else:
        pass

def forYou():
    st.title('Enter your user id')
    user_list = pickle.load(open('user_list.pkl','rb'))
    input = st.text_input('Enter user id')
    if st.button("Recommend"):
        response = requests.get('http://127.0.0.1:8000/user_based?usr_id={}&num_rec={}'.format(input,10))
        data = ast.literal_eval(response.content.decode('UTF-8'))
        data = pd.DataFrame(data)
        data = pd.merge(data,movies,on='movieId')
        data = pd.merge(data,links,on='movieId')
        data.drop('title_y',axis=1,inplace=True)
        data.rename(columns={'title_x':'title'},inplace=True)
        display_movs(data)
        # st.table(data)

def display_movs(data):
    try:
        posters = []
        movie_names = []
        for key in range(len(data)):
            movie_names.append(data['title'][key])
            try:
                response = requests.get('http://127.0.0.1:8000/images/{}'.format(int(data['tmdbId'][key])))
                link = ast.literal_eval(response.content.decode('UTF-8'))
                posters.append(link)
            except:
                link = "Not found"
                posters.append(link)
                
        col1,col2,col3 = st.columns(3)
        col4,col5,col6 = st.columns(3)
        col7,col8,col9 = st.columns(3)
        col10,col11,col12 = st.columns(3)
        with col1:
            st.markdown(movie_names[0])
            st.image(posters[0])
        with col2:
            st.markdown(movie_names[1])
            st.image(posters[1])
        with col3:
            st.markdown(movie_names[2])
            st.image(posters[2])
        with col4:
            st.markdown(movie_names[3])
            st.image(posters[3])
        with col5:
            st.markdown(movie_names[4])
            st.image(posters[4])
        with col6:
            st.markdown(movie_names[5])
            st.image(posters[5])
        with col7:
            st.markdown(movie_names[6])
            st.image(posters[6])
        with col8:
            st.markdown(movie_names[7])
            st.image(posters[7])
        with col9:
            st.markdown(movie_names[8])
            st.image(posters[8])
        with col10:
            st.markdown(movie_names[9])
            st.image(posters[9])
        with col11:
            st.markdown(movie_names[10])
            st.image(posters[10])
        with col12:
            st.markdown(movie_names[11])
            st.image(posters[11])
    except:
        pass

def main():
    st.title("Movie Recommender System")
    menu = ["Popular Movies","Search","For You Page"]
    choice = st.sidebar.selectbox("Menu",menu)
    if choice == 'Popular Movies':
        home()
    elif choice == 'Search':
        search()
    elif choice == 'For You Page':
        forYou()

if __name__ == "__main__":
    main()
