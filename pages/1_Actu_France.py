###############  LES IMPORTS  ###############

import pickle
from pathlib import Path

import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from wordcloud import WordCloud
from PIL import Image
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

import re
import string

# pour les stopwords
import spacy
# Ici on récupère le pipeline français
import spacy.cli

from spacy.lang.fr.stop_words import STOP_WORDS as stopWordFR

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

###############  LA CONFIG DE BASE  ###############

st.set_page_config( page_title = "Actualités",
                    page_icon = "chart_with_upwards_trend",
                    layout="wide")


st.set_option('deprecation.showPyplotGlobalUse', False)

###############   USER AUTHENTIFICATION  ##################

names = ["David Gillard", "Mikael Roor"]
usernames = ["dgillard", "mroor"]

file_path = Path(__file__).parent / "../hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)


authenticator = stauth.Authenticate(names, usernames, hashed_passwords, '1_Actu_France', "abcdef", cookie_expiry_days=0)

name, authentication_status, username = authenticator.login("login", "main")

if authentication_status == False:
    st.error('Erreur de Username/Password')

if authentication_status == None:
    st.warning("Veuillez entrer un username et un password")

if authentication_status:
    authenticator.logout("logout", "sidebar")
    st.sidebar.title(f"Welcome {name}")

    ###############  MISE EN CACHE DU MODEL  ###############

    # cela permet de gagner en rapidité
    @st.cache(allow_output_mutation=True)
    def load_model(model_name):
        spacy.cli.download(model_name)
        nlp = spacy.load(model_name)
        return (nlp)

    #spacy.load("fr_core_news_lg")
    #spacy.load("fr_core_news_md")
    #spacy.load("fr_core_news_sm")
    nlpFR = load_model("fr_core_news_sm")

    ###############  MA LISTE PERSO DES STOPWORDS  ###############
    # Cette liste est à faire évoluer en fonction du besoin
    french_stopwords = [
            "a", "à","actu", "afin", "alors", "ans", "après", "au","aux", "aucuns", "aura", "aussi", "autre", "avant", "avec", "avoir", "beau", "bel", "belle", "bon", "car", "ce", "cela", "ces", "c'est", "ceux", "celle", "cet", "cette","cest", "chaque", "chez", "ci", "comme",
            "comment", "dans","d", "d'", "de", "des", "dun","dune", "d'un", "d'une", "du", "dû", "dedans", "dehors", "depuis", "deux", "devrait", "doit", "donc", "dont", "dos", "début", "elle", "elles", "en", "entre",
            "encore", "enfin", "essai", "est", "et", "eu", "faire", "fait", "faites","fémin", "fois", "font", "grand", "hors", "ici", "il", "ils", "je","juste", "l", "l'", "la", "le", "les", "leur", "là",
            "l'on", "ma", "maintenant", "mais", "mes", "miss", "mien", "moins", "mon", "mot", "même","n", "n'", "ne", "n'est", "ni", "nommés", "notre", "nous", "nouveau", "on", "ont", "ou", "où", "par", "parce",
            "pas", "peut", "peu","plus", "plupart", "pour", "pourquoi", "près","qu", "quand", "que", "quel", "quelle", "quelles", "quels", "qui", "qu'il", "qu'ils", "sa", "saint", "sans","se", "sera", "ses", "s'est",
            "seul","seulement", "si", "sien", "son", "sont", "sous", "soyez", "sujet", "sur", "ta", "tandis", "tellement", "tels", "tes", "ton", "tous",
            "tout", "trop", "très", "tu","un", "une", "va", "voient", "vont", "votre", "vous", "vu", "ça","c'était", "étaient", "état", "étions", "été", "être", "y", "paris",
            "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche", "janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre",
            "novembre", "décembre",
            "pari", "loir", "iledefrance", "ledefrance", "doub", "l'ain","val de", "de france", "et loire", "belfort", "lyon", "france", "fêt", "côteor","e circonscription",
            'ain','aisne',"aisn", 'allier','alpesdehauteprovence', "alpesdehauteprovenc", 'hautesalpes',"hautesalpe",'alpesmaritimes','ardeche','ardèche', 'ardennes','ariege','aube','aude','aveyron','bouchesdurhone','calvados','cantal','charente','charentemaritime','cher','correze',
            'corsedusud','hautecorse','cotedor', 'côtedor', "côte d'or",'cotesdarmor','creuse','dordogne','doubs','drome', 'drôme', 'eure','eureetloir','finistere','gard','hautegaronne','gers','gironde','herault','ileetvilaine','indre','indreetloire','isere', 'isère',
            'jura','landes','loiretcher','loire','hauteloire',"hauteloir", 'loireatlantique','loiret','lot','lotetgaronne','lozere','maineetloire','manche','marne','hautemarne','mayenne','meurtheetmoselle','meuse','morbihan','moselle','nievre',
            'nord','oise','orne','pasdecalais','puydedome','pyreneesatlantiques','hautespyrenees','pyreneesorientales','basrhin','hautrhin','rhone', 'rhône','hautesaone','saoneetloire','sarthe','savoie','hautesavoie','paris','seinemaritime',
            'seineetmarne','yvelines',"yveline",'deuxsevres','somme', "somm",'tarn','tarnetgaronne',"tarnetgaronn",'var','vaucluse',"vauclus",'vendee','vend','vienne',"vienn",'hautevienne',"hautevienn",'vosges','yonne',"yonn",'territoiredebelfort','essonne',"essonn",'hautsdeseine',"hautsdeseine",
            'seinesaintdenis','valdemarne','valdoise', "val d'oise", "valdois", "valdemarn",
            "fte", "tour", "chteaulavallière", "saintnicolasdebourgueil", "savignésurlathan", "langeai", "saintroch", "chouzésurloire",  "cinqmarslapile", "membrollesurchoisille",
            "nordtouraine", "sain","homme", "saint", "dindreetloir", "coteauxsurloire", "tourain", "beaumontlouestault", "cyr", "mazièresdetouraine", "saintantoinedurocher", "neuvyleroi"
        ]


    ###############   FONCTIONS COMMUNES A TOUTES LES PAGES  ##################

    # WORDCLOUD
    # Déclaration de la fonction :
    def data_processing(data):
        #lowercase conversion
        data = data.lower()
        # tokenize the data
        data_tokens = word_tokenize(data)
        # remove stopwords
        processed_words = [w for w in data_tokens if not w in stopWordFR]
        # remove sotpwords with my franch_sotpwords list
        processed_words = [w for w in processed_words if not w in french_stopwords ]
        return " ".join(processed_words)

    # WORDCLOUD
    def plot_cloud(wordcloud):
        plt.figure(figsize=(5,5), facecolor='none')
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.show()

    # NETTOYAGE DES MOTS
    def nettoyage_txt(original_txt, show=False):
        txt = original_txt
        txt = txt.lower() # txt en minuscule
        txt = re.sub('@','',txt) # supprimer @
        txt = re.sub("l'","",txt) # supprimer l'
        txt = re.sub("d'","",txt) # supprimer d'
        txt = re.sub('\[.*\]','',txt) # supprimer le contenu entre crochets
        txt = re.sub('<.*?>+','',txt) # supprimer le contenu entre <>
        txt = re.sub('https?://\S+|www\.\S+', '', txt) # supprimer URLs
        txt = re.sub(re.escape(string.punctuation), '', txt) # supprimer la ponctuation
        txt = re.sub(r'[^a-zA-Zéèëôêâç ]+', '', txt) # supprimer les nombres
        txt = re.sub('\n', '', txt) # supprimer les saut de ligne
        txt = str(txt).strip() # supprimer tous les caractères au début et à la fin de la chaine de caractère
        if show:
            print('text original : ', original_txt)
            print('text nettoyé : ', txt)
        return txt

    # TOKENIZER DE MOTS
    def tokenizeStr(original):
        txt2 = nlpFR(original) # créer une liste de mots
        txt2 = [token.lemma_ for token in txt2 if not nlpFR.vocab[token.text].is_stop]
        punct = string.punctuation
        stopwords = list(stopWordFR)
        ws = string.whitespace
        txt2 = [word for word in txt2 if word not in stopwords and word not in punct if len(word)>2]
        txt2 = [word for word in txt2 if not word in french_stopwords ]
        return txt2

    # SENTIMENT
    # on récupère le model d'algo de VADER
    readSentiment = SentimentIntensityAnalyzer()
    def get_sentiments(text):
        scores = readSentiment.polarity_scores(text)
        return scores.get('compound')
    # on instancie des values
    sentiments = ['Negative', 'Positive', 'Neutral']
    def getSentiment(phrase):
        s = readSentiment.polarity_scores(phrase)
        if s['compound'] <= -0.05:
            sentiment = 0
        elif s['compound'] >= 0.05:
            sentiment = 1
        else:
            sentiment = 2
        return sentiment, s

    ###############    FIN FONCTIONS   ##################


    ###############  TITRE ET SIDEBAR   ###############

    st.markdown(""" <style> .font {
                font-size:55px ; font-family: 'Cooper Black'; color: #FF9633;}
                </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Dashboard des actualités France</p>', unsafe_allow_html=True)
    st.sidebar.markdown("# Actu France")
    st.markdown("## Veuillez importer un fichier .xlsx (excel)")


    ############### IMPORT DU FICHIER   ###############

    #df = pd.read_excel('media_info.xlsx', sheet_name = 0)

    ############### UPLOADED DU FICHIER   ###############

    @st.cache(allow_output_mutation=True)
    def load_data(file):
        df =  pd.read_excel(file, sheet_name = 0)
        return df

    left_column , right_column = st.columns([3,1])
    with left_column:
        uploaded_file = st.file_uploader("Choisir un fichier excel")

    if uploaded_file is not None:
        df = load_data(uploaded_file)


        ############### FILTRE PAR DATE   ###############

        st.markdown("## Veuillez définir la plage de dates")
        # petit probleme en comparant une date de dataframe à une date variable
        # c'est pourquoi il faut ajouter le .dt.date
        df['date'] =  pd.to_datetime(df['date']).dt.date

        # Using current time
        ini_time_for_now = datetime.now()

        #today = datetime.date.today()
        past_date_before_7days = ini_time_for_now - timedelta(days = 7)
        #tomorrow = today + datetime.timedelta(days=1)
        left_column , right_column = st.columns([3,1])
        with left_column:
            start_date = st.date_input('Choisir date de début :', past_date_before_7days)
            #end_date = st.date_input('End date', tomorrow)
            end_date = st.date_input('Choisir date de fin :', ini_time_for_now)
        if start_date > end_date:
            st.error('Error: Date de fin doit être choisi après la date de début.')
        else:
            #greater than the start date and smaller than the end date
            mask = (df['date'] > start_date) & (df['date'] <= end_date)
            df = df.loc[mask]


            ###############  SOUS-TITRE ET DATAFRAME  ###############

            st.markdown("##")
            st.header("Analyse des mots")
            st.subheader("Les données :")
            # CSS to inject contained in a string
            hide_dataframe_row_index = """
                <style>
                    row_heading.level0 {display:none;}
                    blank {display:none;}
                </style>
            """
            # Inject CSS with Markdown
            st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)
            st.dataframe(df)


            ###############   WORDCLOUD   ###############
            st.subheader('Les mots les plus employés')
            titre_df = list(df.titre)
            titre_clean = [nettoyage_txt(titre) for titre in titre_df]
            df['titre_clean'] = titre_clean
            data_titre = df['titre_clean'].apply(data_processing)
            text_titre = ' '.join([w for w in data_titre])
            mask = np.array(Image.open('images/carte_france.png'))
            wordcloud = WordCloud(background_color="ivory", mask=mask).generate(text_titre)
            wd = plot_cloud(wordcloud)
            left_column , middle_column, right_column = st.columns([1,6,1])
            with middle_column:
                st.pyplot(wd)

            ############### TOP 20 DES MOTS   ###############
            df['titre_clean'] = titre_clean
            df['titre_clean'] = df['titre_clean'].apply(data_processing)
            wordsT = [word for i in range(0, len(df)-1) for word in tokenizeStr(df.iloc[i].titre_clean) if str(word).strip() != '']
            wordlist = pd.value_counts(wordsT)
            topW = pd.DataFrame(data={'tag': wordlist.index, 'count':wordlist.values})
            fig = px.bar(topW[:20], y='tag', x='count', orientation='h',
                color='tag', width = 800 ,  height = 600)
            st.subheader("Liste de mots les plus utilisés")
            st.write("Top 20 :")
            left_column , middle_column, right_column = st.columns([1,6,1])
            with middle_column:
                st.plotly_chart(fig)


            ############### CALCUL DE SENTIMENT   ###############
            st.header("Les Sentiments sur les titres concernant la France métropolitaine")

            english_sentiment = []
            english_title_clean = list(df.titre_anglais)

            for txt in english_title_clean:
                english_sentiment.append(getSentiment(txt)[0])

            df['sentiment'] = english_sentiment

            df_sentiment = df[['sentiment','titre_anglais']].groupby('sentiment').count()
            df_sentiment['sentiment_analyse'] = sentiments

            df_sentiment['sentiment_analyse'] = df_sentiment['sentiment_analyse'].astype(str)

            df_sentiment.rename(columns = {'titre_anglais' : 'Nb_Article'}, inplace = True)
            df_sentiment.rename(columns={'sentiment_analyse' : 'Sentiment'}, inplace = True)

            ### Le PIE ###
            st.write("Les chiffres :")
            st.dataframe(df_sentiment)
            total_articles = df_sentiment.Nb_Article.sum()
            st.write(f"Total Articles : {total_articles}")
            colors = ['red', 'green', 'bleu' ]
            fig = go.Figure(
                    go.Pie(
                        labels = df_sentiment.Sentiment,
                        values =  df_sentiment.Nb_Article
                        )
                    )
            fig.update_traces(hoverinfo='label+value', textinfo='percent', textfont_size=20,
                            marker=dict(colors=colors, line=dict(color='#000000', width=2)))
            left_column , middle_column, right_column = st.columns([1,6,1])
            with middle_column:
                st.plotly_chart(fig)

            ##########  LINE_PLOT  ###########
            # un peu de tratement de tableau pour pouvoir afficher le lineplot dans l'analyse positive et négative en dessous colonne de gauche

            dfPivot = df.copy()
            dfPivot = dfPivot.iloc[:,[1,7]]
            conditionlist = [
                (dfPivot['sentiment'] == 0) ,
                (dfPivot['sentiment'] == 1),
                (dfPivot['sentiment'] == 2)]
            choicelist = ['Négatif', 'Positif', 'Neutre']

            dfPivot['libelle'] = np.select(conditionlist, choicelist, default='Not Specified')

            table = dfPivot[['date','libelle', 'sentiment']].groupby(['date', 'libelle']).count()

            table = pd.pivot_table(table, values='sentiment', index=['date'], columns=['libelle'])

            table = table.rename_axis('date').reset_index()

            # on converti les na en 0
            table = table.fillna(0)

            # changement de type
            table.date = table.date.astype('str')
            table.Neutre = table.Neutre.astype('int')
            table.Négatif = table.Négatif.astype('int')
            table.Positif = table.Positif.astype('int')

            count_table = table.Neutre + table.Négatif + table.Positif
            table['neutre_pourcentage'] = ((table.Neutre/count_table)*100)
            table['negatif_pourcentage'] = ((table.Négatif/count_table)*100)
            table['positif_pourcentage'] = ((table.Positif/count_table)*100)
            st.header("Table récapitulative total et pourcentage des sentiments par date")
            st.dataframe(table)

            ############### ANALYSE TITRE POSITIF   ###############
            st.header("Titres dont le sentiment est : Positif")

            df_joy_mask = df['sentiment'] == 1
            joy_filtered_df = pd.DataFrame(df[df_joy_mask])
            joy_filtered_df.sentiment.value_counts()

            df_titre_joy = list(joy_filtered_df.titre)
            titre_joy_clean = [nettoyage_txt(titre) for titre in df_titre_joy]
            joy_filtered_df['titre_joy_clean'] = titre_joy_clean
            joy_filtered_df['titre_joy_clean'] = joy_filtered_df['titre_joy_clean'].apply(data_processing)

            # Dataframe
            st.dataframe(joy_filtered_df.titre)

            fig = plt.figure(figsize=(10, 4))
            sns.lineplot(x = "date", y = "positif_pourcentage", data = table, color="green")
            st.subheader("Pourcentage du sentiment Positif sur le total des articles par date")
            left_column , middle_column, right_column = st.columns([1,6,1])
            with middle_column:
                st.pyplot(fig)

            # bar_chart
            wordsT = [word for i in range(0, len(joy_filtered_df)-1) for word in tokenizeStr(joy_filtered_df.iloc[i].titre_joy_clean) if str(word).strip() != '']
            wordlist = pd.value_counts(wordsT)
            topW = pd.DataFrame(data={'tag': wordlist.index, 'count':wordlist.values})
            fig = px.bar(topW[:20], y='tag', x='count', orientation='h',
                    title='Top 20 mots', color='tag', width = 800 ,  height = 600)
            left_column , middle_column, right_column = st.columns([1,6,1])
            with middle_column:
                st.plotly_chart(fig)

            # Wordcloud
            joy_df = joy_filtered_df['titre_joy_clean'].apply(data_processing)
            titre_joy = ' '.join([w for w in joy_df])
            wordcloud = WordCloud(background_color="ivory").generate(titre_joy)
            wd = plot_cloud(wordcloud)
            st.pyplot(wd)


            ############### ANALYSE TITRE NEGATIF   ###############
            st.header("Titres dont le sentiment est : Négatif")
            df_bad_mask = df['sentiment'] == 0
            bad_filtered_df = pd.DataFrame(df[df_bad_mask])
            bad_filtered_df.sentiment.value_counts()
            df_titre_bad = list(bad_filtered_df.titre)
            titre_bad_clean = [nettoyage_txt(titre) for titre in df_titre_bad]
            bad_filtered_df['titre_bad_clean'] = titre_bad_clean
            bad_filtered_df['titre_bad_clean'] = bad_filtered_df['titre_bad_clean'].apply(data_processing)

            st.dataframe(bad_filtered_df.titre)

            fig = plt.figure(figsize=(10, 4))
            sns.lineplot(x = "date", y = "negatif_pourcentage", data = table, color="red")
            st.subheader("Pourcentage du sentiment Négatif sur le total des articles par date")
            left_column , middle_column, right_column = st.columns([1,6,1])
            with middle_column:
                st.pyplot(fig)

            # Bar_chart
            wordsT = [word for i in range(0, len(bad_filtered_df)-1) for word in tokenizeStr(bad_filtered_df.iloc[i].titre_bad_clean) if str(word).strip() != '']
            wordlist = pd.value_counts(wordsT)
            topW = pd.DataFrame(data={'tag': wordlist.index, 'count':wordlist.values})
            fig = px.bar(topW[:20], y='tag', x='count', orientation='h',
                    title='Top 20 mots', color='tag', width = 800 ,  height = 600)
            left_column , middle_column, right_column = st.columns([1,6,1])
            with middle_column:
                st.plotly_chart(fig)

            # Wordcloud
            bad_df = bad_filtered_df['titre_bad_clean'].apply(data_processing)
            titre_bad = ' '.join([w for w in bad_df])
            wordcloud = WordCloud(background_color="ivory").generate(titre_bad)
            wd = plot_cloud(wordcloud)
            st.pyplot(wd)



# -- Ajout de style ---
st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
        """

st.markdown(st_style, unsafe_allow_html=True)