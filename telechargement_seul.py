#!/usr/bin/env python
# coding: utf-8

# ## Importation des packages

# In[1]:


# -*- coding: utf-8 -*-

import os
import shutil
import numpy as np
import datetime
import json 
import tempfile
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from osgeo import gdal
import fiona
import requests
from pprint import pprint
import rasterio
import rasterio.mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import time
import urllib3
import sys
from PIL import Image, ImageStat
from IPython.display import Markdown, display
import argparse
import json
import os
import pprint
import sys
import traceback
import warnings
import platform
import zipfile

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

## Choix de l'année de traitement
annee = str(datetime.date.today().year)


# ## Dossier

# In[4]:


# Dossier principal
Dossier= f"path/Download{annee}"
if not os.path.isdir(Dossier):
    os.mkdir(Dossier)

# Dossier pour les images découpés et rééchantillonnées
images = f"{Dossier}/images/"
if not os.path.isdir(images):
    os.mkdir(images)  

# Dossier des saisons
ete = f'{Dossier}/images/ete/'
if not os.path.isdir(ete):
    os.mkdir(ete)  

automne = f'{Dossier}/images/automne/'
if not os.path.isdir(automne):
    os.mkdir(automne) 
    
hiver = f'{Dossier}/images/hiver/'
if not os.path.isdir(hiver):
    os.mkdir(hiver) 


# ## Téléchargement

# In[19]:


def telechargement_sentinel2(feature_image) :
    ### Téléchargement d'une image définit par le paramètre feat_img issu d'une requète ###

    print(" ### Début du téléchargement ### ")
    # Début du téléchargement
    start = time.time()
    lien = feature_image ["properties"]["services"]["download"]["url"]
    print(lien)
    # Connexion au serveur
    headers = {'Authorization': 'Bearer %s' % token,}
    params = (('issuerId', 'theia'),)
    img = requests.get(lien, headers=headers, params=params, verify=False, stream=True)
    print(img)
    token
    print("   ################  Token récupéré      ################  ")
    # Si la connexion s'est faite correctement (if img.ok)  on lance le téléchargement
    if img.ok :
        total_size = int(img.headers.get('content-length', 0))
        nom_image = feature_image["properties"]["productIdentifier"]
        print("Nom de l'image : %s" % nom_image , completionDate)
        print("Lien de téléchargement : %s" % lien)  
        print("Taille de l'image : %d" % total_size ," octets")
        
#         On copie l'image par partie de 8 x 1024 bits 
        with open(f"{Dossier}/{nom_image}.zip", "wb") as fd :
            print("Téléchargement en cours...")
            for chunk in img.iter_content(1024 * 8):
                fd.write(chunk)
        
        # Calcul du temps de téléchargement
        diff_time = time.time() - start
        minutes = diff_time // 60
        seconds = diff_time % 60
        print("Image téléchargée en %dmin %ds\n" % (minutes, seconds))
        
    else:
        print("Problème avec la requête.\n")


# ## Dézippage et découpage des images

# In[26]:


def decoupage(fzip, nom_image, sauvegarde):
    
    f = zipfile.ZipFile(fzip)
    with fiona.open(f"path/zone_etude.shp") as shapefile:
        geoms = [feature["geometry"] for feature in shapefile]
    
    if not os.path.isdir(f"{sauvegarde}{nom_image[0:19]}"):
        os.mkdir(f"{sauvegarde}{nom_image[0:19]}")
    
    file = [name for name in f.namelist() if name.endswith('CLM_R1.tif')] 
    print(file)
    with rasterio.open(f"zip://{fzip}!/{file[0]}", masked=True) as clm_all:
        out_image, out_transform = rasterio.mask.mask(clm_all, geoms, crop=True)
        
        out_meta = clm_all.meta.copy()
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})
           
    with rasterio.open(f"{sauvegarde}/{nom_image[0:19]}/{nom_image[0:19]}_CLM_R1_clipped.tif", "w", **out_meta) as dest:
        dest.write(out_image)
        
    print("Découpage bande nuages ok")
    
    # Pour b2
    file = [name for name in f.namelist() if name.endswith('FRE_B2.tif')] 
    with rasterio.open(f"zip://{fzip}!/{file[0]}", masked=True) as b2_all:        
        out_image, out_transform = rasterio.mask.mask(b2_all, geoms, crop=True)
        
        out_meta = b2_all.meta.copy()
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})
        
    with rasterio.open(f"{sauvegarde}/{nom_image[0:19]}/{nom_image[0:19]}_B2_clipped.tif", "w", **out_meta) as dest:
        dest.write(out_image)
    print("Découpage bande B2 ok")
    
    productsb2 = b2_all = out_image = out_transform = out_meta = dest = None
    
    # Pour b3
    file = [name for name in f.namelist() if name.endswith('FRE_B3.tif')] 
    with rasterio.open(f"zip://{fzip}!/{file[0]}", masked=True) as b3_all:
        out_image3, out_transform3 = rasterio.mask.mask(b3_all, geoms, crop=True)
        
        productsb3 = out_image3  
        
        out_meta3 = b3_all.meta.copy()
        out_meta3.update({"driver": "GTiff",
                          "height": out_image3.shape[1],
                          "width": out_image3.shape[2],
                          "transform": out_transform3})
       
    with rasterio.open(f"{sauvegarde}/{nom_image[0:19]}/{nom_image[0:19]}_B3_clipped.tif", "w", **out_meta3) as dest:
        dest.write(productsb3)
    print("Découpage bande B3 ok")

    productsb3 = b3_all = out_image3 = out_transform3 = out_meta3 = dest = None
    
    # Pour b4
    file = [name for name in f.namelist() if name.endswith('FRE_B4.tif')] 
    with rasterio.open(f"zip://{fzip}!/{file[0]}", masked=True) as b4_all:
        out_image4, out_transform4 = rasterio.mask.mask(b4_all, geoms, crop=True)
        
        productsb4 = out_image4
        
        out_meta4 = b4_all.meta.copy()
        out_meta4.update({"driver": "GTiff",
                          "height": out_image4.shape[1],
                          "width": out_image4.shape[2],
                          "transform": out_transform4})
       
    with rasterio.open(f"{sauvegarde}/{nom_image[0:19]}/{nom_image[0:19]}_B4_clipped.tif", "w", **out_meta4) as dest:
        dest.write(productsb4)
    print("Découpage bande B4 ok")

    productsb4 = b4_all = out_image4 = out_transform4 = out_meta4 = dest = None
    
    # Pour b5
    file_zip = [name for name in f.namelist() if name.endswith('FRE_B5.tif')] 
    print(file_zip)
    f.extract(file_zip[0], path = f"{sauvegarde}/{nom_image[0:19]}/")
    for file in glob(f"{sauvegarde}{nom_image[0:19]}/{file_zip[0][0:49]}*B5.tif"):
        i = file
        print(i)
        resample=os.path.join(i, i.replace(".tif","_resample.tif"))
        gdal.Warp(resample, i, xRes= 10 , yRes= 10)
        print("Resample ok")
        with rasterio.open(resample, masked=True) as b5_all:
            out_image5, out_transform5 = rasterio.mask.mask(b5_all, geoms, crop=True)
            
            productsb5 = out_image5
            
            out_meta5 = b5_all.meta.copy()
            out_meta5.update({"driver": "GTiff",
                              "height": out_image5.shape[1],
                              "width": out_image5.shape[2],
                              "transform": out_transform5})
       
        with rasterio.open(f"{sauvegarde}/{nom_image[0:19]}/{nom_image[0:19]}_B5_clipped.tif", "w", **out_meta5) as dest:
            dest.write(productsb5)
            
    shutil.rmtree(f"{sauvegarde}{nom_image[0:19]}/{file_zip[0][0:49]}")    
    print("Découpage et resample bande B5 ok")

    productsb5 = b5_all = out_image5 = out_transform5 = out_meta5 = dest = None

    # Pour b6
    file_zip = [name for name in f.namelist() if name.endswith('FRE_B6.tif')]
    f.extract(file_zip[0], path = f"{sauvegarde}/{nom_image[0:19]}/")
    for file in glob(f"{sauvegarde}{nom_image[0:19]}/{file_zip[0][0:49]}*B6.tif"):
        i=file
        print(i)
        resample=os.path.join(i,i.replace(".tif","_resample.tif"))
        gdal.Warp(resample, i, xRes= 10 , yRes= 10)
        print("Resample ok")
        with rasterio.open(resample, masked=True) as b6_all:
            out_image6, out_transform6 = rasterio.mask.mask(b6_all, geoms, crop=True)
            
            productsb6 = out_image6
            
            out_meta6 = b6_all.meta.copy()
            out_meta6.update({"driver": "GTiff",
                              "height": out_image6.shape[1],
                              "width": out_image6.shape[2],
                              "transform": out_transform6})
       
        with rasterio.open(f"{sauvegarde}/{nom_image[0:19]}/{nom_image[0:19]}_B6_clipped.tif", "w", **out_meta6) as dest:
            dest.write(productsb6)
        
    shutil.rmtree(f"{sauvegarde}{nom_image[0:19]}/{file_zip[0][0:49]}")       
    print("Découpage bande B6 ok")
    
    productsb6 = b6_all = out_image6 = out_transform6 = out_meta6 = dest = None
    
    # Pour b7
    file_zip = [name for name in f.namelist() if name.endswith('FRE_B7.tif')] 
    f.extract(file_zip[0], path = f"{sauvegarde}/{nom_image[0:19]}/")
    for file in glob(f"{sauvegarde}{nom_image[0:19]}/{file_zip[0][0:49]}*B7.tif"):
        i=file
        print(i)
        resample=os.path.join(i,i.replace(".tif","_resample.tif"))
        gdal.Warp(resample, i, xRes= 10 , yRes= 10)
        print("Resample ok")        
        with rasterio.open(resample, masked=True) as b7_all:
            out_image7, out_transform7 = rasterio.mask.mask(b7_all, geoms, crop=True)
            
            productsb7 = out_image7
            
            out_meta7 = b7_all.meta.copy()
            out_meta7.update({"driver": "GTiff",
                              "height": out_image7.shape[1],
                              "width": out_image7.shape[2],
                              "transform": out_transform7})
       
        with rasterio.open(f"{sauvegarde}/{nom_image[0:19]}/{nom_image[0:19]}_B7_clipped.tif", "w", **out_meta7) as dest:
            dest.write(productsb7)
    
    shutil.rmtree(f"{sauvegarde}{nom_image[0:19]}/{file_zip[0][0:49]}")    
    print("Découpage bande B7 ok")

    productsb7 = b7_all = out_image7 = out_transform7 = out_meta7 = dest = None

    # Pour b8
    file = [name for name in f.namelist() if name.endswith('FRE_B8.tif')]
    with rasterio.open(f"zip://{fzip}!/{file[0]}", masked=True) as b8_all:
        out_image8, out_transform8 = rasterio.mask.mask(b8_all, geoms, crop=True)
        
        productsb8 = out_image8
        
        out_meta8 = b8_all.meta.copy()
        out_meta8.update({"driver": "GTiff",
                          "height": out_image8.shape[1],
                          "width": out_image8.shape[2],
                          "transform": out_transform8})
       
    with rasterio.open(f"{sauvegarde}/{nom_image[0:19]}/{nom_image[0:19]}_B8_clipped.tif", "w", **out_meta8) as dest:
        dest.write(productsb8)
    print("Découpage bande B8 ok")

    productsb8 = b8_all = out_image8 = out_transform8 = out_meta8 = dest = None
    
    # Pour b8a
    file_zip = [name for name in f.namelist() if name.endswith('FRE_B8A.tif')] 
    f.extract(file_zip[0], path = f"{sauvegarde}/{nom_image[0:19]}/")
    for file in glob(f"{sauvegarde}{nom_image[0:19]}/{file_zip[0][0:49]}*B8A.tif"):
        i=file
        print(i)
        resample=os.path.join(i,i.replace(".tif","_resample.tif"))
        gdal.Warp(resample, i, xRes= 10 , yRes= 10)
        print("Resample ok")
        with rasterio.open(resample, masked=True) as b8a_all:
            out_image8a, out_transform8a = rasterio.mask.mask(b8a_all, geoms, crop=True)
            
            productsb8a = out_image8a
            
            out_meta8a = b8a_all.meta.copy()
            out_meta8a.update({"driver": "GTiff",
                               "height": out_image8a.shape[1],
                               "width": out_image8a.shape[2],
                               "transform": out_transform8a})
       
        with rasterio.open(f"{sauvegarde}/{nom_image[0:19]}/{nom_image[0:19]}_B8A_clipped.tif", "w", **out_meta8a) as dest:
            dest.write(productsb8a)
    
    shutil.rmtree(f"{sauvegarde}{nom_image[0:19]}/{file_zip[0][0:49]}")    
    print("Découpage bande B8A ok")
    
    # Pour b11
    file_zip = [name for name in f.namelist() if name.endswith('FRE_B11.tif')] 
    f.extract(file_zip[0], path = f"{sauvegarde}/{nom_image[0:19]}/")
    for file in glob(f"{sauvegarde}{nom_image[0:19]}/{file_zip[0][0:49]}*B11.tif"):
        i=file
        print(i)
        resample=os.path.join(i,i.replace(".tif","_resample.tif"))
        gdal.Warp(resample, i, xRes= 10 , yRes= 10)
        print("Resample ok")
        with rasterio.open(resample, masked=True) as b11_all:
            out_image11, out_transform11 = rasterio.mask.mask(b11_all, geoms, crop=True)
            
            productsb11 = out_image11
            
            out_meta11 = b11_all.meta.copy()
            out_meta11.update({"driver": "GTiff",
                               "height": out_image11.shape[1],
                               "width": out_image11.shape[2],
                               "transform": out_transform11})
       
        with rasterio.open(f"{sauvegarde}/{nom_image[0:19]}/{nom_image[0:19]}_B11_clipped.tif", "w", **out_meta11) as dest:
            dest.write(productsb11)
    
    shutil.rmtree(f"{sauvegarde}{nom_image[0:19]}/{file_zip[0][0:49]}")
    print("Découpage bande B11 ok")
    
    # Pour b12
    file_zip = [name for name in f.namelist() if name.endswith('FRE_B12.tif')] 
    f.extract(file_zip[0], path = f"{sauvegarde}/{nom_image[0:19]}/")
    for file in glob(f"{sauvegarde}{nom_image[0:19]}/{file_zip[0][0:49]}*B12.tif"):
        i=file
        print(i)
        resample=os.path.join(i,i.replace(".tif","_resample.tif"))
        gdal.Warp(resample, i, xRes= 10 , yRes= 10)
        print("Resample ok")
        with rasterio.open(resample, masked=True) as b12_all:
            out_image12, out_transform12 = rasterio.mask.mask(b12_all, geoms, crop=True)
            
            productsb12 = out_image12
            
            out_meta12 = b12_all.meta.copy()
            out_meta12.update({"driver": "GTiff",
                               "height": out_image12.shape[1],
                               "width": out_image12.shape[2],
                               "transform": out_transform12})
       
        with rasterio.open(f"{sauvegarde}/{nom_image[0:19]}/{nom_image[0:19]}_B12_clipped.tif", "w", **out_meta12) as dest:
            dest.write(productsb12)

    shutil.rmtree(f"{sauvegarde}{nom_image[0:19]}/{file_zip[0][0:49]}")
    print("Découpage bande B12 ok")
    
# file = i = b2_all = b3_all = b4_all = b5_all = b6_all = b7_all = b8_all = b8a_all = b11_all = b12_all = clm_all = dest = None


# ## Requête téléchargement site theia du cnes

# In[21]:


## Récupération du token
config = "path/config_theia.cfg"
df = pd.read_csv(config, sep="=", header=None, names=["Cle", "Valeur"])
df.Cle = df.Cle.str.strip() # Suppression des espaces avant et après le texte
df.Valeur = df.Valeur.str.strip()

token = requests.post("https://theia.cnes.fr/atdistrib/services/authenticate/",
                      data={"ident": df.Valeur[df.Cle == "login_theia"],
                            "pass": df.Valeur[df.Cle == "password_theia"]},
                      verify=False).text
if token == '':
    print('Pas de token recu ou probleme de connexion ! ')
    sys.exit(1)


# In[22]:


# Paramètres des images sentinel à télécharger
params = {"collection": "SENTINEL2",
            "location": "T30TWT",
            "processingLevel" : "LEVEL2A",
            "relativeOrbitNumber" : "137",
             
            "startDate" : annee+"-01-01",  
            "completionDate": annee+"-12-30", 
            "maxRecords" : 500,} 


# In[58]:


url = "https://theia.cnes.fr/atdistrib/resto2/api/collections/SENTINEL2/search.json"
req = requests.get(url, verify=False, params=params)
    
if req.ok:
    req_images = json.loads(req.text)
    # On ne garde que la clé features qui nous interesse
    feat_img = req_images["features"]
    # Suppression de tags inintéressants et très longs avant affichage
    for image in feat_img:
        del image["properties"]["keywords"]
        del image["properties"]["links"]
    print("Nombre d'image %d\n" % len(feat_img))
        #print(feat_img)
else:
    print("Problème avec la requête.")

################## Affinage de la requète

with open('datas.json', 'w') as f:
    f.write(req.text)
    
################## Affichage de quelques infos par image contenues dans le fichier json

i=0
liste = []
dates = [f"{annee}01", f"{annee}02", f"{annee}06", f"{annee}07", 
         f"{annee}08", f"{annee}10", f"{annee}11"]

for f in feat_img:
    
    prod = f["properties"]["productIdentifier"]
    print("i :",i)
    print("prod :",prod)
    
    # Téléchargement des images qui nous intéresse pour les traitements 
    if any(x in prod for x in dates):
    
        cloudCover = f["properties"]["cloudCover"]
        platform = f["properties"]["platform"]
        processingLevel = f["properties"]["processingLevel"]
        version = f["properties"]["version"]
        feature_id = f["id"]
        completionDate = f["properties"]["completionDate"]
        download = f["properties"]["services"]["download"]["url"]
        orbit=f["properties"]["relativeOrbitNumber"]
        liste.append(i)

    i+=1    
    
    
print("numero des images avec moins de 30% de nuage ", liste) 
nb_images = len(feat_img)
print("Nombre d'images correpondant à la requête : ", len(liste)," sur un total de ",len(feat_img))    


# # Lancement du script

# In[62]:


print ("             ### Lancement du téléchargement ###             ")
img = 0
# for i, img in enumerate(liste):
while img<len(liste):
    telechargement_sentinel2(feat_img[liste[img]])
    token = requests.post("https://theia.cnes.fr/atdistrib/services/authenticate/",
                      data={"ident": df.Valeur[df.Cle == "login_theia"],
                            "pass": df.Valeur[df.Cle == "password_theia"]},
                          verify=False).text

    fzip = [name for name in glob(f"path{annee}/**zip")][0]
    name = fzip[len(f"path{annee}/"):]
    
    print ("             ### Lancement du découpage ###             ")
    
    # Pour le dossier été
    if f"{annee}06" in fzip or f"{annee}07" in fzip or f"{annee}08" in fzip:
        sauvegarde = ete
        decoupage(fzip, name, ete)
    
    # Pour le dossier automne
    elif f"{annee}10" in fzip or f"{annee}11" in fzip:
        sauvegarde = automne
        decoupage(fzip, name, automne)
    
    # Pour le dossier hiver
    elif f"{annee}01" in fzip or f"{annee}02" in fzip or f"{annee}12" in fzip:
        decoupage(fzip, name, hiver)
    
    os.remove(fzip)
    img += 1

