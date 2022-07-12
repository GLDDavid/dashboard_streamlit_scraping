###############  LES INSTALL  ###############
# install modules dans un command prompt
#pip install requests
#pip install bs4
#pip install pandas
#pip install schedule
#pip install vaderSentiment
#pip install translators --upgrade

###############  LES IMPORTS  ###############

import pickle
from pathlib import Path

import streamlit as st
import streamlit_authenticator as stauth

from urllib.request import Request , urlopen
from bs4 import BeautifulSoup
from datetime import date
import time
import pandas as pd
import os
import sqlite3
import translators as ts
import random, requests
from lxml import html


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


authenticator = stauth.Authenticate(names, usernames, hashed_passwords, '4_Scraper_France', "abcdef", cookie_expiry_days=0)

name, authentication_status, username = authenticator.login("login", "main")

if authentication_status == False:
    st.error('Erreur de Username/Password')

if authentication_status == None:
    st.warning("Veuillez entrer un username et un password")

###############   USER AUTHENTIFICATION OK  ##################
if authentication_status:
    authenticator.logout("logout", "sidebar")
    st.sidebar.title(f"Welcome {name}")

    st.markdown(""" <style> .font {
            font-size:55px ; font-family: 'Cooper Black'; color: #FF9633;}
            </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Récupération d\'informations concernant l\'actualité France</p>', unsafe_allow_html=True)

###############   CREATION DES LISTES  ##################
    dept_test = ['Manche',"Indre-et-Loire"]

    # liste des départements
    depts = ["Ain","Aisne","Allier","Alpes-de-Haute-Provence","Hautes-Alpes","Alpes-Maritimes","Ardeche","Ardennes","Ariege","Aube","Aude","Aveyron",
    "Bouches-du-Rhone","Calvados","Cantal","Charente","Charente-Maritime","Cher","Correze","Corse-du-Sud","Haute-Corse","Cote-d'Or",
    "Cotes-d'Armor","Creuse","Dordogne","Doubs","Drome","Eure","Eure-et-Loir","Finistere","Gard","Haute-Garonne","Gers","Gironde",
    "Herault","Ile-et-Vilaine","Indre","Indre-et-Loire","Isere","Jura","Landes","Loir-et-Cher","Loire","Haute-Loire","Loire-Atlantique",
    "Loiret","Lot","Lot-et-Garonne","Lozere","Maine-et-Loire","Manche","Marne","Haute-Marne","Mayenne","Meurthe-et-Moselle","Meuse",
    "Morbihan","Moselle","Nievre","Nord","Oise","Orne","Pas-de-Calais","Puy-de-Dome","Pyrenees-Atlantiques","Hautes-Pyrenees","Pyrenees-Orientales",
    "Bas-Rhin","Haut-Rhin","Rhone","Haute-Saone","Saone-et-Loire","Sarthe","Savoie","Haute-Savoie","Paris","Seine-Maritime","Seine-et-Marne","Yvelines",
    "Deux-Sevres","Somme","Tarn","Tarn-et-Garonne","Var","Vaucluse","Vendee","Vienne","Haute-Vienne","Vosges","Yonne","Territoire-de-Belfort",
    "Essonne","Hauts-de-Seine","Seine-Saint-Denis","Val-de-Marne","Val-d'Oise"]


    deptsAtoC = ["Ain","Aisne","Allier","Alpes-de-Haute-Provence","Hautes-Alpes","Alpes-Maritimes","Ardeche","Ardennes","Ariege","Aube","Aude","Aveyron",
    "Bouches-du-Rhone","Calvados","Cantal","Charente","Charente-Maritime","Cher","Correze","Corse-du-Sud","Haute-Corse","Cote-d'Or",
    "Cotes-d'Armor","Creuse"]
    deptsDtoL = ["Dordogne","Doubs","Drome","Eure","Eure-et-Loir","Finistere","Gard","Haute-Garonne","Gers","Gironde",
    "Herault","Ile-et-Vilaine","Indre","Indre-et-Loire","Isere","Jura","Landes","Loir-et-Cher","Loire","Haute-Loire","Loire-Atlantique",
    "Loiret","Lot","Lot-et-Garonne","Lozere"]
    deptsMtoH = ["Maine-et-Loire","Manche","Marne","Haute-Marne","Mayenne","Meurthe-et-Moselle","Meuse",
    "Morbihan","Moselle","Nievre","Nord","Oise","Orne","Pas-de-Calais","Puy-de-Dome","Pyrenees-Atlantiques","Hautes-Pyrenees","Pyrenees-Orientales",
    "Bas-Rhin","Haut-Rhin"]
    deptsRtoV = ["Rhone","Haute-Saone","Saone-et-Loire","Sarthe","Savoie","Haute-Savoie","Paris","Seine-Maritime","Seine-et-Marne","Yvelines",
    "Deux-Sevres","Somme","Tarn","Tarn-et-Garonne","Var","Vaucluse","Vendee","Vienne","Haute-Vienne","Vosges","Yonne","Territoire-de-Belfort",
    "Essonne","Hauts-de-Seine","Seine-Saint-Denis","Val-de-Marne","Val-d'Oise"]

###############   DATE  ##################
    # On récupère la date du jour
    today = date.today().strftime("%Y-%m-%d")

###############   BDD  ##################
    # Création de la base de données
    def f_creerLaBaseDeDonnees():
        if os.path.isfile('base_scraping.db' ):
            print("la base de données existe déjà." )
            st.info("La base de données existe déjà.")
        else :
            connexion = sqlite3.connect("base_scraping.db" )
            curseur = connexion.cursor()
            curseur.execute("""
                CREATE TABLE scraper (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                jour CURRENT VARCHAR(10) NOT NULL,
                departement VARCHAR(25) NOT NULL,
                titre VARCHAR(150) NOT NULL,
                titre_anglais VARCHAR(150) NOT NULL,
                media VARCHAR(50) NOT NULL
                );
                """ )
            print("bdd ok")
            st.success("La Base de données a été créée")
            connexion.commit()
            connexion.close()

###############   DEBUT SCRAPING  ##################
    st.subheader("Sélection des départements")
    titre = []
    departement = []
    english_title = []

    headers= ""

    depts_selected = st.selectbox(
        'Choisir les départements',
        (deptsAtoC, deptsDtoL, deptsMtoH, deptsRtoV, dept_test))

    left_column , right_column = st.columns([3,1])
    with left_column:
        st.info(f"Votre Sélection: {depts_selected}")
        st.warning("Si code erreur 429, c'est que trop de requêtes ont été effectuée. Il faut donc recommencer demain.")

    st.subheader("Scraping")
    if st.button('Démarrer le scraping'):
        medias = []
        f_creerLaBaseDeDonnees()
        st.info(f"On est dans la boucle et on a sélectionné :  {depts_selected}")
        for i in depts_selected:
            st.info(i)
            # Céation de plusieurs user_agent et je boucle dessus pour changer à chaque requête
            user_agent_list = [
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.1 Safari/605.1.15',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:77.0) Gecko/20100101 Firefox/77.0',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:77.0) Gecko/20100101 Firefox/77.0',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
            ]

            for _ in range(len(user_agent_list)):
                # Random sur user agent
                user_agent = random.choice(user_agent_list)
                headers = {'User-Agent': user_agent}

            # url avec le département et la date du jour
            #https://news.google.com/search?q=ain%202022-07-01&hl=fr&gl=FR&ceid=FR%3Afr
            #https://news.google.com/search?q=ain%202022-07-01&hl=fr&gl=FR&ceid=FR%3Afr
            url = f"https://news.google.com/search?q={i}%20{today}&hl=fr&gl=FR&ceid=FR%3Afr"

            st.info(f"Le User-Agent pour cette requête est : {headers}")

            st.info(f"L'url de la requête : {url}")

            req = Request(url, headers=headers)

            webpage = urlopen(req).read()

            soup = BeautifulSoup(webpage, 'lxml')

            # boucle dans la soup pour récupérer toutes les infos (titres)
            infos = soup.find_all('a', attrs={'class' : 'DY5T1d RZIKme'})
            for info in infos:
                a = info.get_text()
                titre.append(a)

            # on va maintenant traduire le titre
                english_title.append(ts.google(a))

            # on envoie aussi les départements ici
                departement.append(i)

            # la liste des médias avec le xpath
            connexion = requests.get(url)
            # Je transforme l'objet en HTML
            page_html = html.fromstring(html=connexion.text)
            # je récupère tous les médias et je transforme tout ça en liste
            media = page_html.xpath('//*[@id="yDmH0d"]//div/main/c-wiz//div[1]/a/text()')
            medias += media
            #print(medias)

            # un petit time pour faire une pause de 1 seconde avant de relancer la requete sur une autre url
            time.sleep(1)
        st.success("Scraping terminé")
###############   FIN SCRAPING  ##################

###############   DATAFRAME  ##################
        # Création du DataFrame
        df_medias = pd.DataFrame({
            "jour": today,
            "departement" : departement,
            "titre" : titre,
            "titre_anglais": english_title,
            "media" : medias
        })
        st.dataframe(df_medias)

        # on supprime les doublons
        df_medias.drop_duplicates(subset ="titre", keep = 'first', inplace=True)

        # Création d'une liste de tupples à partir des valeurs du dataframe
        tpls = [tuple(x) for x in df_medias.to_numpy()]
        #print(tpls)

        # Sauvegarde du dataframe en base de données SQLITE
        bdd = sqlite3.connect("base_scraping.db" )
        curseur = bdd.cursor()
        curseur.executemany('INSERT INTO scraper (jour, departement, titre, titre_anglais, media) values(?,?,?,?,?)',tpls)
        bdd.commit()
        curseur.close()

        # Pour info :
        #  le temps d'exécution de cette cellule et de plus ou moins 60 minutes en fonction du débit et du nombres d'articles.
    else:
        pass

###############   RECUPERATION DES DONNEES BDD EN FICHIER EXCEL  ##################
    st.subheader("Dataframe to excel")
    st.write("On peut maintenant exporter les données dans un fichier excel")
    if st.button('Création fichier excel'):
            ## on test une requete
            ## bloc de code a effectué dans un bloc de code différent
            bdd = sqlite3.connect("base_scraping.db")
            curseur = bdd.cursor()
            requete = "SELECT * FROM scraper;"
            curseur.execute(requete)
            resultats = curseur.fetchall()

            # je récupère les données pour analyse
            nom_colonne = ['id', 'date','departement', 'titre', 'titre_anglais', 'media']
            df_result = pd.DataFrame(data=resultats, columns=nom_colonne)

            # on ferme la connexion
            curseur.close()

            # on donne un nom au fichier
            filename = 'import_export_fichier/media_info2.xlsx'

            # writing to Excel
            file_result = pd.ExcelWriter(filename)

            # on sauvegarde le fichier en xlsx
            df_result.to_excel(file_result, index=False, header=True)

            file_result.save()

            st.success("Le fichier a bien été créé")
    else:
        pass

