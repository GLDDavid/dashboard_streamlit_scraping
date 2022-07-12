# Install de base
#python -m venv venv
#python -m pip install --upgrade pip
#pip install streamlit
#pip install --upgrade streamlit

# Install analyse de mots
# pip install openpyxl   (pour ouvrir un fichier excel)
# pip install xlrd       (toujours pour les fichiers xl)
# pip install nltk (pour l'analyse des mots avec par exemple word_tokenize)

# Install représentation graphique ou autre
# pip install wordcloud (nuage de mots)
# pip install matplotlib
# pip install plotly

# Pour le traitement des mots
# pip install -U spacy

# VADER
# pip install vaderSentiment

# Pour l'authentification
# pip install streamlit_authenticator

# pour les animations lottie
# pip install streamlit-lottie


import pickle
from pathlib import Path

import json


import streamlit as st
import streamlit_authenticator as stauth
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie


# je customize la page
st.set_page_config( page_title = "Actualités",
                        page_icon = "chart_with_upwards_trend",
                        layout="wide")

###############   USER AUTHENTIFICATION  ##################

names = ["David Gillard", "Mikael Roor"]
usernames = ["dgillard", "mroor"]

file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

authenticator = stauth.Authenticate(names, usernames, hashed_passwords, 'Accueil', "abcdef", cookie_expiry_days=0)

name, authentication_status, username = authenticator.login("login", "main")

if authentication_status == False:
    st.error('Erreur de Username/Password')

if authentication_status == None:
    st.warning("Veuillez entrer un username et un password")

if authentication_status:

    ###############   HEADER  ##################

    authenticator.logout("logout", "sidebar")
    st.sidebar.title(f"Welcome {name}")

    st.sidebar.markdown("# Accueil")

    col1, col2 = st.columns( [0.8, 0.2])
    with col1:               # To display the header text using css style
        st.markdown(""" <style> .font {
            font-size:55px ; font-family: 'Cooper Black'; color: #FF9633;}
            </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">L\'analyse de données concernant l\'actualités</p>', unsafe_allow_html=True)
    with col2:               # To display brand log
            #st.image(logo, width=130 )
        st.write("Mon logo ici")

    st.write("Cette application va vous permettre de comprendre les différentes tendances de l'actulalités française.")
    st.write("Vous allez pouvoir visualiser le résultat d'analyse concernant les départements mais également les différentes circonscriptions.")


    ###############   LOTTIE ANIMATION  ##################
    def load_lottiefile(filepath : str):
        with open(filepath, 'r') as f:
            return json.load(f)
    lottie_data_analysis = load_lottiefile("lottiefiles/data-analysis.json")
    left_column ,middle_column, right_column = st.columns([1,5,1])
    with middle_column:
        st_lottie(
            lottie_data_analysis,
            speed = 1,
            reverse = False,
            loop=True,
            quality="medium", # low ; high
            height=None,
            width=None,
            key=None
        )





# -- Ajout de style ---
st_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
    """

st.markdown(st_style, unsafe_allow_html=True)


