import pickle
import pandas as pd
import streamlit as st
from urllib.request import urlopen
from bs4 import BeautifulSoup
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

cv = CountVectorizer(max_features=5000 , stop_words='english')


def display(name, poster):
    col0, col1, col2, col3, col4 = st.columns(5)
    with col0:
        st.text(name[0])
        st.image(poster[0])

    with col1:
        st.text(name[1])
        st.image(poster[1])

    with col2:
        st.text(name[2])
        st.image(poster[2])

    with col3:
        st.text(name[3])
        st.image(poster[3])

    with col4:
        st.text(name[4])
        st.image(poster[4])
    col5, col6, col7, col8, col9 = st.columns(5)

    with col5:
        st.text(name[5])
        st.image(poster[5])

    with col6:
        st.text(name[6])
        st.image(poster[6])

    with col7:
        st.text(name[7])
        st.image(poster[7])

    with col8:
        st.text(name[8])
        st.image(poster[8])

    with col9:
        st.text(name[9])
        st.image(poster[9])


# Anime
def fetching_poster_anime(anime_id):
    htmldata = urlopen(f'https://myanimelist.net/anime/{anime_id}')
    soup = BeautifulSoup(htmldata, 'html.parser')
    images = soup.find_all('img')
    link = images[1]['data-src']
    return link

@st.experimental_memo
def recommend_anime(title):
    idx = anime[anime['name'] == title].index[0]
    sim_scores = list(enumerate(anime_similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    recd_anime = []
    recd_anime_poster = []
    for i in sim_scores:
        anime_id = anime.iloc[i[0]]['anime_id']
        recd_anime.append(anime.iloc[i[0]]['name'])
        recd_anime_poster.append(fetching_poster_anime(anime_id))
    return recd_anime, recd_anime_poster




#Books
def recommend_books(title):
    idx = np.where(books_pt.index == title)[0][0]
    sim_scores = list(enumerate(books_similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    recd_books = []
    recd_books_poster = []
    for i in sim_scores:
        recd_books.append(books_pt.index[i[0]])
        poster = books[books['Book-Title'] == books_pt.index[i[0]]]['Image-URL-M'].values[0]
        recd_books_poster.append(poster)
    return recd_books, recd_books_poster

# Movies
def recommend_movies(title):
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(movies_similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]

    recd_movies = []
    recd_movies_poster = []
    for i in sim_scores:
        recd_movies.append(movies.iloc[i[0]].title)
        recd_movies_poster.append(movies.iloc[i[0]].poster)
    return recd_movies, recd_movies_poster

#Music

def fetching_music_poster(song_name):
    sp.search(q=song_name, type='track')
    url = sp.search(q=song_name, type='track')['tracks']['items'][0]['album']['images'][0]['url']
    return url

@st.experimental_memo
def recommend_music(song_name):
    index = music[music['song_name'] == song_name].index[0]
    sim_scores = list(enumerate(music_similarity[index]))
    sim_scores = sorted(sim_scores, key=lambda x:x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    song_indices = [i[0] for i in sim_scores]
    recd_music = []
    recd_music_poster = []
    for i in song_indices:
        recd_music.append(music.iloc[i]['song_name'])
        recd_music_poster.append(fetching_music_poster(music.iloc[i]['song_name']))
    return recd_music, recd_music_poster

if "Select" not in st.session_state:
    st.session_state.button_clicked = False


def callback():
    # Button was clicked!
    st.session_state.button_clicked = True


st.title("Welcome to Recommendation System By Rah's")

selection = st.selectbox(
    'Select The section You want to be recommended',
    ('--Select Option--','Anime', 'Books', 'Movies', 'Music'))


if selection == 'Anime':
    gengre_list = pickle.load(open('genre_list.pkl', 'rb'))
    vector = cv.fit_transform(gengre_list).toarray()
    anime_similarity = cosine_similarity(vector)
    anime_dict = pickle.load(open('Anime.pkl', 'rb'))
    anime = pd.DataFrame(anime_dict)
    Anime = st.selectbox(
        'select a Anime you like', anime['name'].values)

    recommend_anime, recommend_anime_poster = recommend_anime(Anime)
    display(recommend_anime, recommend_anime_poster)

elif selection == 'Books':
    books = pickle.load(open('books.pkl', 'rb'))
    books = pd.DataFrame(books)
    books_pt = pickle.load(open('books_pt.pkl', 'rb'))
    books_pt = pd.DataFrame(books_pt)
    books_similarity = pickle.load(open('book_similarity.pkl', 'rb'))
    
    Books = st.selectbox(
        'select a Book you like', books_pt.index)

    recommend_books, recommend_books_poster = recommend_books(Books)
    display(recommend_books, recommend_books_poster)



elif selection == 'Movies':
    tags = pickle.load(open('tags.pkl', 'rb'))
    vector = cv.fit_transform(tags).toarray()
    movies = pickle.load(open('movies.pkl', 'rb'))
    movies_similarity = similarity = cosine_similarity(vector)
    movie = st.selectbox(
        'select a movie you like', movies['title'].values)

    recommend_movies, recommend_movies_poster = recommend_movies(movie)
    display(recommend_movies, recommend_movies_poster)

elif selection == 'Music':
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="fe6b07ee1b0e4481876512cb332425aa",
                                                           client_secret="a6a46012ac3e41af83a85f09d628c7d1"))
    music_dict = pickle.load(open('music.pkl', 'rb'))
    music = pd.DataFrame(music_dict)
    music_similarity = pickle.load(open('Music_similarity.pkl', 'rb'))
    Music = st.selectbox(
        'select a music you like', music['song_name'].values)

    recommend_music, recommend_music_poster = recommend_music(Music)
    display(recommend_music, recommend_music_poster)