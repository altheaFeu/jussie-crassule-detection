#!/usr/bin/env python
# coding: utf-8

# # Installation des packages

# In[20]:


#!/usr/bin/env python

import gc-
import pysptools
import pysptools.abundance_maps as amp
import numpy as np
from joblib import Parallel, delayed
import os
from numpy import genfromtxt
from numpy.linalg import multi_dot
import matplotlib.pyplot as plt
from osgeo import gdal
from gdalconst import GA_ReadOnly
import rasterio
import rasterio.mask
from rasterio.windows import Window
from rasterio.transform import Affine
from rasterio.mask import mask
from glob import glob
import pandas as pd
import joblib
from pprint import pprint
import datetime
import time
import geopandas as gpd
import cvxopt
import fiona
import shutil
import rasterio
import geopandas as gpd
import rioxarray as rxr
from shapely.geometry import box
from shapely.geometry import mapping

annee = str(datetime.date.today().year)


# ## Enregistrement des images

# In[2]:


def create_tif(Dossier_save, name_save, image,raster_test):
    
    os.chdir(Dossier_save)
    image_int=image

    # Récupération des informations sur l'image de référence
    im_ref = gdal.Open(raster_test, GA_ReadOnly)
    projection = im_ref.GetProjection()
    geotransform = im_ref.GetGeoTransform()
    ncol = im_ref.RasterXSize
    nrow = im_ref.RasterYSize
    im_ref = None
    print ("Infos recup")
    
    # Enregistrement de l'image de démélengeage
    driver = gdal.GetDriverByName( "GTiff" )
    test=driver.Create(name_save +".tif",ncol, nrow, 1, gdal.GDT_Int16)
    test.SetProjection( projection )
    test.SetGeoTransform(geotransform )
    test.GetRasterBand(1).WriteArray( image_int )
    test = projection = geotransform = None
    print("   ###   Sauvegarde demelange effectuee   ###   ")


# ## Fonctions de démélengeage

# In[3]:


# Fonction pour le démélengeage
def sma_jussie(data,U, le_mask, dossier_resultat):
    fcls = amp.FCLS() #Application d'un "Fully Constrained Least Square"
    print("démélange en cours")
    u1 = fcls.map(data, U, normalize=False, mask = le_mask) 
    
    fcls.plot(dossier_resultat, colorMap='jet', suffix='demel',columns=3)
    return u1
    
# Récupération du spectre de démélengeage et copie du résultat
def demelangeage(data_test, data_msk, dossier, dossier_sentinel, dossier_resultat):

    #### spectre en réflectance csv changer le chemin 
    
    spectre = genfromtxt("/home/afeuillet/Documents/Data/spectre_final.csv", delimiter=',')

    result=sma_jussie(data_test,spectre, data_msk, dossier_resultat)
    
    print("demel fini, ecriture raster")
    
###On peut rajouter d'autres résultats de la librairie, result[:,:,0] coresspond à la première ligne de la librairie 
    result_j=result[:,:,0]
    result_j = result_j*1000
    
    result_c = result[:,:,1]
    result_c = result_c*1000

    nom = dossier_sentinel[0:19]
    
    b2 = f"{dossier}/{dossier_sentinel}_B2_clipped.tif"
    create_tif(dossier_resultat,f"{nom}_fcls_jussie", result_j,b2)
    create_tif(dossier_resultat,f"{nom}_fcls_crassule", result_c,b2)
    
    result_j = result_c = b2 = spectre = result = None


# ## Création du stack
# In[4]:


def stack_jussie(dossier):
    os.chdir(dossier)
    
    for filename in glob(dossier + "*B2_clipped.tif"):
        print(filename)
        with rasterio.open(filename) as src_data:
            data=src_data.read()
            data_b2=data[0,:,:]
            print(data_b2.shape)
        print("ok b2")
        
    for filename in glob(dossier + "*B3_clipped.tif"):
        print(filename)
        with rasterio.open(filename) as src_data:
            data=src_data.read()
            data_b3=data[0,:,:]
            print(data_b3.shape)
        print("ok b3")
        
    for filename in glob(dossier + "*B4_clipped.tif"):
        print(filename)
        with rasterio.open(filename) as src_data:
            data=src_data.read()
            data_b4=data[0,:,:]
            print(data_b4.shape)
        print("ok b4")
        
    for filename in glob(dossier + "*B5_clipped.tif"):
        print(filename)
        with rasterio.open(filename) as src_data:
            data=src_data.read()
            data_b5=data[0,:,:]
            print(data_b5.shape)
        print("ok b5")

    for filename in glob(dossier + "*B6_clipped.tif"):
        print(filename)
        with rasterio.open(filename) as src_data:
            data=src_data.read()
            data_b6=data[0,:,:]
            print(data_b6.shape)
        print("ok b6")


    for filename in glob(dossier + "*B11_clipped.tif"):
        print(filename)
        with rasterio.open(filename) as src_data:
            data=src_data.read()
            data_b11=data[0,:,:]
            print(data_b11.shape)
        print("ok b11")

    for filename in glob(dossier + "*B12_clipped.tif"):
        print(filename)
        with rasterio.open(filename) as src_data:
            data=src_data.read()
            data_b12=data[0,:,:]
            print(data_b12.shape)
        print("ok b12")
        
### ICI, LA BOUCLE TOURNE BIEN MAIS A LA FIN, TOUS LES FICHIERS CREES SONT SUPPRIMES

    data_stack=np.stack((data_b2,data_b3,data_b4,data_b5,data_b6,data_b11,data_b12))
    
    # Problèmes de mémoire
    data_b2 = 0
    data_b3 = 0
    data_b4 = 0
    data_b5 = 0
    data_b6 = 0
    data_b11 = 0
    data_b12 = 0
    
    data_stack=data_stack/10000

    data_stack=np.moveaxis(data_stack,0,-1)

    print("stack ok")
    return(data_stack)


# ## Calcul du pourcentage des nuages

# In[5]:


def nuage(chemin, img):
    
    ## Importation de la bande des nuages
    with rasterio.open(f"{chemin}/{img}/{img}_CLM_R1_clipped.tif") as src_nuage:
        data_nuage=src_nuage.read(1)
        
        ### Caclul pourcentage de nuage
        unique, count = np.unique(data_nuage, return_counts = True)
        raster_dataframe = pd.DataFrame()
        raster_dataframe['Pixel Value']=unique
        raster_dataframe['Count']=count

        sum_total = raster_dataframe['Count'].sum()
        count = 0
        
        ### Recherche des valeurs égales à 0
        if 0 in raster_dataframe['Pixel Value'].values:
            count = count + float(raster_dataframe.loc[raster_dataframe['Pixel Value']==0, 'Count']) 
        
        ### Recherche des valeurs égales à 2
        if 2 in raster_dataframe['Pixel Value'].values:
            count = float(raster_dataframe.loc[raster_dataframe['Pixel Value']==2, 'Count'])

        pourcentage = 100-((count/sum_total)*100)

        pourcentage = float(pourcentage)
        pourcentage = round(pourcentage, 2)

        print("\n{}\n".format(img))
        print(f"Il y a {pourcentage}% de nuages\n")

        return pourcentage


# ## Masque des prairies

# In[10]:


## Dossiers
dossier_prairies = f"/home/afeuillet/Documents/Data/Download{annee}/images/automne/"
dossier_resultat = f"/home/afeuillet/Documents/Data/Download{annee}/msk_prairies/"

if not os.path.exists(dossier_resultat): os.makedirs(dossier_resultat)
nom_automne = []
for dossier in os.listdir(dossier_prairies) : nom_automne.append(dossier)
    
pourcent_final = 0
nom_a_supprimer = []

for i, img in enumerate(nom_automne):
    # Calcul du pourcentage
    pourcentage = nuage(dossier_prairies, img)
    
    if pourcentage < 30:
        nom_a_supprimer.append(img)
        # Création du masque
        with rasterio.open(f"{dossier_prairies}/{img}/{img}_B3_clipped.tif") as src_data:
            data_automne = src_data.read()
            data_automne = np.where((data_automne <= 500),1,0)
            
        data_final = data_automne

        out_meta = src_data.meta.copy()
        out_meta.update({"driver": "GTiff",
                        "height": data_automne.shape[1],
                        "width": data_automne.shape[2],
                        "transform": src_data.transform,
                        'dtype':'int16',
                        'nodata':0})

        with rasterio.open(f"{dossier_resultat}/{img}_msk_prairies.tif", "w", **out_meta) as dest:
            dest.write(data_automne)       
        
        # Calcul du pourcentage de zones sans prairies
        with rasterio.open(f"{dossier_resultat}/{img}_msk_prairies.tif") as src:
            array = src.read()

        unique, count = np.unique(array, return_counts = True)
        raster_dataframe = pd.DataFrame()
        raster_dataframe['Pixel Value']=unique
        raster_dataframe['Count']=count
        print(raster_dataframe)

        sum_count = raster_dataframe['Count'].sum()

        nb = float(raster_dataframe.loc[raster_dataframe['Pixel Value']==1, 'Count'])

        print(sum_count)
        pourcent = (nb/sum_count)*100

        pourcent = float("{0:.2f}".format(pourcent))
        print(f"La zone d'inondation du {img[11:19]} a {100-pourcent}% de prairies")    
        
        # Conservation de la zone avec le moins de prairies
        if pourcent_final < pourcent:
            nom_automne_final = img
            pourcent_final = pourcent 
            

print(f"L'image sélectionné du {nom_automne_final} a {100-pourcent_final}% de zones de prairies")

# Suppression de toutes les masques de prairies sauf celle conservée pour le démélengeage
nom_a_supprimer.remove(nom_automne_final)

for i, img in enumerate(nom_a_supprimer):os.remove(f"{dossier_resultat}/{img}_msk_prairies.tif")


# ## Fonction lancement

# In[11]:


def lancement(dossier_sentinel):
    
    # Dossiers de travail
    base==f"path{annee}/images/ete_conserve/"
    date=dossier_sentinel[11:19]
    
    dossier_resultat= f"path{annee}/Demel_{annee}"
    if not os.path.exists(dossier_resultat):os.makedirs(dossier_resultat)
    
    dossier=f"{base}/{dossier_sentinel}/"
    msk = "path/asmk"
    
    data_test = stack_jussie(dossier)
    
    print("trt des masques")
    
    ## Masque des nuages
    for nuage in glob(f"{dossier}{dossier_sentinel}_CLM_R1_clipped.tif"):
        with rasterio.open(nuage) as src_data:
            data_nuage=src_data.read()
            data_nuage[data_nuage==0]=2
            data_nuage[data_nuage != 2]=0
            data_nuage[data_nuage == 2]=1
    
    ## Masque d'eau
    with rasterio.open(f"path{annee}0101.tif") as src_eau:
        data_eau = src_eau.read()
        data_eau[data_eau!=1]=0
    
    ## Masque des canalisations
    with rasterio.open(f"{msk}/msk_canalisations.tif") as src_canal :
        data_canal = src_canal.read()
        data_canal[data_canal!=1]=0
    
    ## Masque des bocages
    with rasterio.open(f"{msk}/msk_bocages.tif") as src_boc :
        data_boc = src_boc.read()
        data_boc[data_boc!=1]=0

    ## Masque des algues
    with rasterio.open(f"{dossier}{dossier_sentinel}_B2_clipped.tif") as src_algues :
        data_algues = src_algues.read()
        data_algues = np.where((data_algues <= 500),1,0)
    
    ## Masque des prairies
    with rasterio.open(f"path{annee}/msk_prairies/{nom_automne_final}_msk_prairies.tif") as src_prairies :
        data_prairies = src_prairies.read()
        data_prairies[data_prairies!=1]=0
        
    ## Masques totaux
    data_msk=data_nuage*data_eau*data_canal*data_boc*data_algues*data_prairies
    data_msk=data_msk[0,:,:]
    
    ###conversion en boolen pour le masque
    data_msk=data_msk.astype('bool')

    ### pour éviter les problemes de memoire mais gc.collect normalement rend les 4 lignes ci-dessous inutiles
    data_nuage = data_eau = data_canal = data_boc = data_algues = data_prairies = 0
    
    print("debut demelange")
    demelangeage(data_test, data_msk, dossier, dossier_sentinel, dossier_resultat)

    ###evite probleme de memoire
    #gc.collect()
    data_test = None


# ## Dossier

# In[16]:


data_nuages = {'dates':[], 'nuages':[]} #Dictionnaire pour les pourcentages de nuages

base=f"path{annee}/images/ete/"
dossier_travail = f"path{annee}/images/ete_conserve/"

if not os.path.exists(dossier_travail):
    os.makedirs(dossier_travail)
    
dossier_sentinel=[]
for dossier in os.listdir(base) : dossier_sentinel.append(dossier)

## Conservation des images avec moins de 30% de nuage 
for i, img in enumerate(dossier_sentinel): 
    pourcentage = nuage(base, img)
    
    data_nuages['dates'].append(f"{img[11:15]}-{img[15:17]}-{img[17:19]}")
    data_nuages['nuages'].append(pourcentage)
    
    if pourcentage < 30 : 
        shutil.copytree(f"{base}/{img}/",f"{dossier_travail}/{img}/")

dossier_sentinel = []
for dossier in os.listdir(dossier_travail) : dossier_sentinel.append(dossier)


# In[17]:


# Visualisation du pourcentage de nuage
output_path = "output_path"
df_nuages = pd.DataFrame(data_nuages)
df_nuages['dates'] = pd.to_datetime(df_nuages["dates"], format = "%Y-%m-%d", errors = "ignore")
df_nuages = df_nuages.sort_values(by="dates")
date_debut = df_nuages.iloc[0,0]
date_fin = df_nuages.iloc[-1,0]   

# Modélisation
fig, axs = plt.subplots(figsize = (6,4))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
df_nuages.plot(x = "dates", y = "nuages", title = f"Pourcentage de nuage pour l\'annee {annee}", xlabel = "date", ylabel = "%", legend = False, ax=axs, color = "#6D92A0")

# Sauvegarde
fig.savefig(f"{output_path}/pourcentage_nuage_{annee}.png")


# # Lancement du script

# In[21]:


start = time.time()

## Lancement du démélengeage avec 3 cores
Parallel(n_jobs = 3)(delayed(lancement)(img) for img in dossier_sentinel)

print ("             ### Démélengeage Terminé ###             ") 
diff_time = time.time() - start
minutes = diff_time // 60
seconds = diff_time % 60
print(f"Images démélengées en {minutes}min {seconds}s\n")


# In[ ]:




