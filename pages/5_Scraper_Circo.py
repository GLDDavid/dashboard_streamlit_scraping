###############  LES INSTALL  ###############
# pip install pymysql

###############  LES IMPORTS  ###############

import pickle
from pathlib import Path

import streamlit as st
import streamlit_authenticator as stauth

import os
import pandas as pd

import requests
from bs4 import BeautifulSoup as bs
from datetime import date

import random

import pymysql

import translators as ts


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


authenticator = stauth.Authenticate(names, usernames, hashed_passwords, '5_Scraper_Circo', "abcdef", cookie_expiry_days=0)

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
    st.markdown('<p class="font">Récupération d\'informations concernant l\'actualité des circonscriptions</p>', unsafe_allow_html=True)

    os.getcwd()
    print(os.getcwd())
    # C:\Users\Cefim\Desktop\analyse_actu_streamlit

    #Déclaration du chemin du répertoire de travail
    repertoire = "C:/Users/Cefim/Desktop/analyse_actu_streamlit"
    os.chdir(repertoire)

    #Importation des données communes
    communes = pd.read_csv('import_export_fichier/export_dataframe.csv', sep="," , encoding = 'UTF-8', decimal='.')
    st.dataframe(communes)

    commune_test = ['ambillou-37002']

###############   CREATION DOSSIER  ##################
    # Création d'un dossier afin d'enrigistrer toutes les pages html
    st.subheader("Préparation de l'environnement")
    st.write("Création du dossier dans lequel je vais placer les fichiers html scraper")

    def f_creerFichierScrapingHtml():
        if os.path.isfile('import_export_fichier/actu_scraping' ):
            print("Le dossier existe déjà." )
            left_column , right_column = st.columns([3,1])
            with left_column:
                st.info("Le dossier existe déjà.")
        else :
            path = 'import_export_fichier/actu_scraping'
            os.makedirs(path, exist_ok=True)
            print("Le nouveau dossier est créé")
            left_column , right_column = st.columns([3,1])
            with left_column:
                st.info("Le nouveau dossier est créé")

    f_creerFichierScrapingHtml()


###############   DEBUT SCRAPING ET ENREGISTREMENT DES PAGES HTML DANS LE DOSSIER ##################
    st.subheader("Programme de Scraping")
    left_column , right_column = st.columns([3,1])
    with left_column:
        if st.button("Démarrer le Scraping") :
            st.text("Que le scraping commence !!!")
            # on récupère la date du jour
            today = date.today().strftime("%d/%m/%Y")

            # on va faire une copie du html et l'enregistrer
            for i in communes:
            #for i in commune_test:
                url = f"https://www.francebleu.fr/centre-val-de-loire/indre-et-loire-37/{i}"
                st.write(f"Requête sur : {url}")
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
                    st.write(f"Le user-Agent est : {headers}")
                    r = requests.get(url, headers=headers)
                    with open(f'./import_export_fichier/actu_scraping/actu_ville_{i}.html', 'w', encoding="utf-8") as file:
                        file.write(r.text)

                    st.text(f"Ecriture du fichier {i} html => OK")


        ###############   RECUPERATION DES INFOS  ##################

            list_geo = []
            list_ville = []
            list_titre = []
            list_categorie = []
            list_href = []

            # je boucle dans ma liste pour récupérer les infos sur les fichiers dans mon dossier actu_scraping
            for i in communes:
            #for i in commune_test:
                actu = bs(open(f'./import_export_fichier/actu_scraping/actu_ville_{i}.html', encoding="utf-8"), 'html.parser')

                villes = actu.find_all({'span'}, {'class': "rfb-highlight-card-location"})
                list_ville += [span.text.strip() for span in villes]

                titres = actu.find_all({'p'}, {'class' : "rfb-highlight-card-title"})
                list_titre += [p.text.strip() for p in titres]

                depts = actu.find_all({'a'}, {'class' : "Link__UndecoratedLink-sc-198nqgd-0 Item__LinkComponent-sc-1tomiwi-1 eGpVKL"})
                list_geo += [a.text.strip() for a in depts]

                categories = actu.find_all({'a'}, {'class' : "rfb-highlight-card-theme rfb-link"})
                list_categorie += [a.text.strip() for a in categories]

                list_lien = actu.find_all({"a"}, {"class":"rfb-highlight-card-main-link"})
                list_href +=["https://www.francebleu.fr/centre-val-de-loire/indre-et-loire-37" + link['href'].strip() for link in list_lien if link.has_attr('href') and link['href'].startswith(("/emissions","/infos","/vie-quotidienne","/culture","/sports", "/centre-val-de-loire"))]
                list_href
            st.info("Fin de récupération des infos dans les fichiers")

        ###############   TRADUCTION DES TITRES ##################
            list_english_title = []
            # on va maintenant traduire le titre
            for titre in list_titre:
                list_english_title.append(ts.google(titre))

            st.info("La traduction des titres est effectuée")

        ###############   DATAFRAME  ##################
            # on déclare la variable à la date du jour
            date_format = date.today().strftime("%Y/%m/%d")

            # Je récupère le département et la region
            departement = list_geo[2]
            region = list_geo[1]

            # je place tout dans le dataframe
            df_infos = pd.DataFrame({
                        'date' : date_format,
                        'ville' : list_ville,
                        'titre' : list_titre,
                        'titre_anglais' : list_english_title,
                        'categorie': list_categorie,
                        'region' : region,
                        'departement' : departement,
                        'lien' : list_href
            })
            st.success("Le dataframe a été créé")
            st.dataframe(df_infos)

            # je supprime les doublons
            df_infos.drop_duplicates(subset ="titre", keep = 'first', inplace=True)

        ###############   ENREGISTREMENT DATABASE MYSQL  ##################
            st.subheader("Partie SQL")
            st.write("Maintenant que l'on a récupérer les données on va les sauvegarder en base de données")

            st.text("Sauvegarde en cours")
            ###############   DATABASE MYSQL  ##################
            # Enregistrement sur MySQL Workbench
            # Dans MySQL Workbench

            # Creer une database:
            # CREATE DATABASE db_base_actu;

            # on utilise cette base:
            # USE db_base_actu;

            # Creer la table:
            #CREATE TABLE scraper_actu(
            #    id INTEGER PRIMARY KEY AUTO_INCREMENT,
            #    date DATE NOT NULL,
            #    ville VARCHAR(50) NOT NULL,
            #    titre TINYTEXT NOT NULL,
            #    titre_anglais TINYTEXT NOT NULL,
            #    categorie VARCHAR(50) NOT NULL,
            #    region VARCHAR(40) NOT NULL,
            #    departement VARCHAR(50) NOT NULL
            #    lien VARCHAR(255) NOT NULL );

            # on se connecte à notre bdd
            # c'est ici qu'il va falloir changer la connection le jour où on utilise une base de données en ligne
            # Ici je suis en local
            my_conn=pymysql.connect(host='localhost',port=int(3306),user='root',passwd='root',db='db_base_actu')

            Row_list =[]

            # je passe le dataFrame en liste pour l'enregistrer dans sql car le tupple ne fonctionne pas.
            for index, rows in df_infos.iterrows():
                my_list =[rows.date, rows.ville, rows.titre, rows.titre_anglais, rows.categorie, rows.region, rows.departement, rows.lien]
                Row_list.append(my_list)

            # J'insère les infos en base de données
            curseur = my_conn.cursor()
            curseur.executemany('INSERT INTO scraper_actu (date, ville, titre, titre_anglais, categorie, region, departement, lien ) values(%s,%s,%s,%s,%s,%s,%s,%s)',Row_list)
            my_conn.commit()

            # je ferme la connection
            curseur.close()

            print("Le DataFrame a été enregistré en base de données SQL")
            st.info("Le DataFrame a été enregistré en base de données SQL")


            ###############   SUPPRESSION DES DOUBLONS DANS MYSQL  ##################
            # connection à la bdd et on effectue la requête

            # Tout d’abord, on va récupérer la clé primaire la plus petite de chaque groupe de doublons de façon à le garder et effacer les autres
            # Ensuite, on va lier cette sous-requête à notre table principale par la clé primaire pour ne garder que les lignes qui ont soit aucun doublon, soit quand ils ont un doublon, la ligne avec la clé primaire la plus petite.
            # Donc les lignes qui ne seront pas liés seront les lignes à supprimer, les lignes en doublon.

            #bdd = pymysql.connect(host='localhost',port=int(3306),user='root',passwd='root',db='db_base_actu')
            #curseur = bdd.cursor()
            ##requete = "DELETE FROM scraper_actu WHERE id IN (SELECT id FROM (SELECT id, ROW_NUMBER() OVER (PARTITION BY ville, titre) as RowNumber FROM scraper_actu) AS sub WHERE RowNumber > 1);"
            #requete = "DELETE scraper_actu FROM scraper_actu LEFT OUTER JOIN (SELECT MIN( id ) AS id, date, ville, titre, titre_anglais, categorie, region, departement FROM scraper_actu GROUP BY ville, titre) AS scraper_actu_1 ON scraper_actu.id = scraper_actu_1.id WHERE scraper_actu_1.id IS NULL;"
            #curseur.execute(requete)
            #resultats = curseur.fetchall()
            #curseur.close()
            #st.info(f"Suppression des doublons : {resultats}")

            db = pymysql.connect(host='localhost',port=int(3306),user='root',passwd='root',db='db_base_actu')
            # prepare a cursor object using cursor() method
            cursor = db.cursor()
            # Prepare SQL query to DELETE required records
            sql = "DELETE scraper_actu FROM scraper_actu LEFT OUTER JOIN (SELECT MIN( id ) AS id, date, ville, titre, titre_anglais, categorie, region, departement FROM scraper_actu GROUP BY ville, titre) AS scraper_actu_1 ON scraper_actu.id = scraper_actu_1.id WHERE scraper_actu_1.id IS NULL;"
            try:
            # Execute the SQL command
                cursor.execute(sql)
            # Commit your changes in the database
                db.commit()
            except:
            # Rollback in case there is any error
                db.rollback()
            # disconnect from server
            db.close()
            st.info(f"Suppression des doublons")

            ###############   VERIFICATION DE LA SUPPRESSION DES DOUBLONS DANS MYSQL  ##################
            # connection à la bdd et on effectue la requête
            bdd = pymysql.connect(host='localhost',port=int(3306),user='root',passwd='root',db='db_base_actu')
            curseur = bdd.cursor()
            requete = "SELECT COUNT(*) AS nbr_doublon, date, ville, titre, categorie, region, departement FROM scraper_actu GROUP BY ville, titre HAVING COUNT(*) > 1;"
            curseur.execute(requete)
            resultats = curseur.fetchall()
            curseur.close()
            st.write(f"Liste des doublons qui n'ont pas été supprimé : {resultats}")
        else :
            pass


###############   RECUPERATION DES DONNEES ET CREATION DU DATAFRAME  ##################
    st.subheader("Dataframe to excel")
    st.write("On peut maintenant exporter les données dans un fichier excel")
    left_column , right_column = st.columns([3,1])
    with left_column:
        if st.button("Cliquer pour exporter"):
            # connection à la bdd et on effectue la requête
            bdd = pymysql.connect(host='localhost',port=int(3306),user='root',passwd='root',db='db_base_actu')
            curseur = bdd.cursor()
            requete = "SELECT * FROM scraper_actu;"
            curseur.execute(requete)
            resultats = curseur.fetchall()
            df=pd.read_sql(requete,bdd)
            curseur.close()
            st.success("Le dataframe est créé")
            st.dataframe(df)
            # on supprime les doublons
            df.drop_duplicates(subset ="titre", keep = 'first', inplace=True)

        ###############   EXPORT DU DATAFRAME EN FICHIER EXCEL  ##################
            # on donne un nom à notre fichier
            file_name = 'import_export_fichier/actu_circo5_2.xlsx'
            # sauvegarde sous xl
            df.to_excel(file_name, index=False)
            print('Fichier sauvegardé sous excel')
            st.success("Le fichier excel a été créé")
        else:
            pass
