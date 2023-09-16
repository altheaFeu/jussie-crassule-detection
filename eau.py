#!/usr/bin/env python
# coding: utf-8

# In[4]:


#!/usr/bin/env python

import os, sys
from optparse import OptionParser
import fiona
import shutil
import rasterio
import gdal
import datetime
from rasterio.windows import Window
from rasterio.transform import Affine
from rasterio.mask import mask
from glob import glob
import numpy as np
import pandas as pd

annee = str(datetime.date.today().year)


# ## Calcul de l'aire

# In[35]:


def surface(dossier_resultat, plans_eau, result_txt):

    aire_finale = 0

    for i, img in enumerate(plans_eau):
        # Information sur la bande
        ds = gdal.Open(f"{dossier_resultat}/{img}")
        taille_pixel = ds.GetGeoTransform()[1]
        band = ds.GetRasterBand(1)
        array = band.ReadAsArray()

        # Calcul aire de la zone d'inondation
        unique, count = np.unique(array, return_counts = True)
        raster_dataframe = pd.DataFrame()
        raster_dataframe['Pixel Value']=unique
        raster_dataframe['Count']=count

        aire = taille_pixel*taille_pixel*raster_dataframe['Count'][1]*0.0001
        aire = float("{0:.5f}".format(aire))
        print(f"Le plan d\'eau du {img[11:19]} a une aire de {aire} ha")    
        
        # Choix de l'aire finale
        if aire_finale < aire:
            image_finale = img
            aire_finale = aire 

    ds = taille_pixel = band = array = None
    unique = count = raster_dataframe = sum_count = aire = None

    # Enregistrement des informations dans un fichier txt
    print('Annee: {}'.format(annee), file=open(result_txt, "a"))
    print('Image: {} Aire : {}'.format(image_finale, aire_finale), file=open(result_txt, "a"))
    print('===========================================\n', file=open(result_txt, "a"))
    
    # Suppression du reste des plans d'eau
    plans_eau.remove(image_finale)
    for img2 in plans_eau : os.remove(f"{dossier_resultat}{img2}")
        
    return image_finale


# ## Dossier

# In[7]:


base = f"path{annee}/images/hiver/"
msk =  f"path/mask/"
dossier_resultat = f"path{annee}/inondee/"

## Création dossier de sortir
if not os.path.exists(dossier_resultat):os.makedirs(dossier_resultat)

# Liste des images d'hivers
dossier_hiver = []
for folder in os.listdir(base): dossier_hiver.append(folder)
    
print(dossier_hiver)


# ## Lancement du script pour les plans d'eau

# In[9]:


for i, img in enumerate(dossier_hiver):
    
    ## Ouverture bande B8
    with rasterio.open(f"{base}/{img}/{img}_B8_clipped.tif") as src_eau:
        print(f"{base}/{img}/{img}_B8_clipped.tif")
        data_eau=src_eau.read()
        data_eau = np.where((data_eau<=500),1,0)
    
    # Ouverture de la bande des nuages
    with rasterio.open(f"{base}/{img}/{img}_CLM_R1_clipped.tif") as src_nuage:
        data_nuage=src_nuage.read()
        data_nuage = np.where((data_nuage==0),1,0)
    
    ## Ouverture du masque des bocages
    with rasterio.open(f"{msk}/msk_bocages.tif") as src_boc:
        data_boc = src_boc.read()
        data_boc[data_boc==-9999]=0
        
    ## Ouverture du masque des zones salées
    with rasterio.open(f"{msk}/msk_sel.tif") as src_sel:
        data_sel = src_sel.read()
        data_sel[data_sel==-9999]=0

    ## Ouverture du masque des villes associée à la BD topage et aux données OSM
    with rasterio.open(f"{msk}/msk_ville_topage.tif") as src_top:
        data_top = src_top.read()
        data_top[data_top==-9999]=0
         
    data_final = data_eau*data_nuage*data_sel*data_top*data_boc
    
    out_meta = src_eau.meta.copy()
    out_meta.update({"driver": "GTiff",
                    "height": data_final.shape[1],
                    "width": data_final.shape[2],
                    "transform": src_eau.transform,
                    "nodata":0})

    with rasterio.open(f"{dossier_resultat}{img}_plans_eau.tif", "w", **out_meta) as dest:
        dest.write(data_final)
    
    data_ville = data_eau = data_sel = data_boc = data_nuage = None
    src_ville = src_eau = src_sel = src_boc = src_nuage = None


# In[36]:


plans_eau = []
for files in os.listdir(dossier_resultat): plans_eau.append(files)
    
# Fichier pour enregistrer l'aire
result_txt = f"path_result.txt"
image_finale = surface(dossier_resultat, plans_eau, result_txt)


# ## Lancement du script pour les zones inondées

# In[44]:


## Ouverture de la zone d'eau
with rasterio.open(f"{dossier_resultat}/{image_finale}") as src_eau:
    meta_eau = src_eau.meta.copy()
    data_eau=src_eau.read()
    data_eau[data_eau==-9999]=0

## Ouverture du masque des villes associée à la BD topage et aux données OSM
with rasterio.open(f"{msk}/msk_ville_topage.tif") as src_topage:
    data_topage = src_topage.read()
    data_topage[data_topage==-9999]=0


data_final = data_eau+data_topage
data_final[data_final == 2]=1

out_meta = src_eau.meta.copy()
out_meta.update({"driver": "GTiff",
                "height": data_final.shape[1],
                "width": data_final.shape[2],
                "transform": src_eau.transform,
                "nodata":0})

with rasterio.open(f"{dossier_resultat}SENTINEL2A_{annee}0101.tif", "w", **out_meta) as dest:
    dest.write(data_final)
    
with rasterio.open(f"{dossier_resultat}plan_eau_{annee}0201.tif", "w", **meta_eau) as dest:
    dest.write(data_eau)
    
os.remove(f"{dossier_resultat}/{image_finale}")


# ## Tuilage et droits

# In[56]:


# Application du tuilage et de l'overview

script = "path/geotiff/tiff2overview.py"
dpe = "path-plan-eau"
dpi = "path-zone-inondees"

cmd = f'python {script} -i {dossier_resultat}plan_eau_{annee}0201.tif -n 23 -d {dpe} -t 512 -c NONE'
os.system(cmd)
cmd = f'python {script} -i {dossier_resultat}SENTINEL2A_{annee}0101.tif -n 23 -d {dpi} -t 512 -c NONE'
os.system(cmd)

# Suppression du dossier et des images
shutil.rmtree(dossier_resultat)

# Droit d'accès linux
cmd = f'chmod 777 {dpe}/plan_eau_{annee}0201.tif'
os.system(cmd)
cmd = f'chmod 777 {dpi}/SENTINEL2A_{annee}0101.tif'
os.system(cmd)

