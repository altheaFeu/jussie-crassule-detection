#!/usr/bin/env python
# coding: utf-8

# In[49]:


import gc
import pysptools
import pysptools.abundance_maps as amp
import numpy as np
from joblib import Parallel, delayed
import os
from numpy import genfromtxt
from numpy.linalg import multi_dot
import matplotlib.pyplot as plt
import gdal
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
import cvxopt
import geopandas as gpd
import fiona
import rasterio
import shutil
import geopandas as gpd
import rioxarray as rxr
from shapely.geometry import box
from shapely.geometry import mapping

annee = str(datetime.date.today().year)


# ## Données

# In[2]:


base = f"path{annee}/Demel_{annee}/"

ds = []
for folder in os.listdir(base):
    if folder.endswith(".tif"):
        ds.append(folder)


# ## Suppression des valeurs inférieures à 200

# In[3]:


for i, img in enumerate(ds):
    print(img)
    with rasterio.open(f"{base}/{img}") as src:
        data = src.read()
        data = np.where(data >= 200, data, 0)
        
        out_meta = src.meta.copy()
        
        with rasterio.open(f"{base}/{img}", "w", **out_meta) as dest:
            dest.write(data)


# ## Comparaison multi temporelle valeurs totales

# In[38]:


print("comparaison multi date")

#Utilisation pour l'indice de qualité des images
data_ponderation = {'dates':[], 'abondance':[]}
# results_txt = f'Z:/USERS/althea_pnrbriere/Download2020/Demel_2020_fcls/indice_ponderation_{annee}.txt'

######### dossier résultat à changer######################
dossier_final=f"path{annee}/Demel_{annee}/resultats"

if not os.path.exists(dossier_final):
    os.makedirs(dossier_final)

## pour un même site va récuperer les valeurs max issues du démélange (recouvrement max de jussie pour un pixel)

liste_ete=[]

for img in ds:
    with rasterio.open(f"{base}/{img}") as src_data_jussie:
        data_jussie=src_data_jussie.read()
        liste_ete.append(data_jussie)
        
#         Ajout à la pondération
        data_ponderation['dates'].append(f"{img[11:15]}-{img[15:17]}-{img[17:19]}")
        print(f"{img[11:15]}-{img[15:17]}-{img[17:19]}")
        amax = np.amax(data_jussie, axis = 0)
        ele_remove = amax[amax < 200]
        amax = amax[~np.isin(amax, ele_remove)]
        print(amax)
        data_ponderation['abondance'].append(np.max(amax))

st=np.stack(liste_ete)
print(st.shape)

#recupere les pourcentages max de recouvrement toute date confondues
st=np.amax(st,axis=0)
st=st[0,:,:]

with rasterio.open(f"{base}/{ds[0]}") as src:
    out_meta = src.meta.copy()
    with rasterio.open(f"{dossier_final}/demel_{annee}0701.tif", "w", **out_meta) as dest:
        dest.write(st,1)

print("Valeurs conservés pour la pondération\n", data_ponderation)


# ## Pondération

# In[31]:


#Création d'une base d'indices 

## Création d'espace vide pour les 3 parties du graphe
base_debut = pd.Series(pd.date_range(f'{annee}-06-01', periods = 91, freq = 'D'))
base_debut = base_debut.dt.date
base_milieu = base_debut.loc[30:60]
base_fin = base_debut.loc[61:91]

## Reset les indices pour éviter des confusions
base_debut = base_debut.reset_index(drop = True)
base_milieu = base_milieu.reset_index(drop = True)
base_fin = base_fin.reset_index(drop = True)

## On ajoute des valeurs qui vont de 0 à 100 à chacune des parties du graphe
values_increase = pd.Series((np.arange(1,100,3.4)**2)/100)
values_middle = pd.Series(100, index=range(31))
values_decrease = pd.Series((np.arange(100,1,-3.3)**2)/100)

df_debut = pd.concat([base_debut[:30],values_increase], axis = 1)
df_milieu = (pd.concat([base_milieu,values_middle], axis = 1)
             .set_index(np.arange(30,61,1)))
df_fin = (pd.concat([base_fin,values_decrease], axis = 1, ignore_index = True)
          .set_index(np.arange(61,91,1)))

## Modification des colomnes
df_debut.columns = ['dates', 'valeurs'] 
df_milieu.columns = ['dates', 'valeurs']
df_fin.columns = ['dates', 'valeurs']

## Concaténation de tout les dataframes
df_basepd_jussie = pd.concat([df_debut,df_milieu,df_fin], axis = 0)


# In[32]:


#Visualisation base de pondération
fig, axs = plt.subplots(figsize = (10,8))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
df_basepd_jussie.plot(x = "dates", y = "valeurs", title = f"Base de la pondération", xlabel = "date", ylabel = "valeurs", legend = False, ax=axs)
plt.xticks(rotation=50)
plt.show()


# In[39]:


results_txt = "path/indice_ponderation.txt"

# Transformation
data_ponderation = pd.DataFrame(data_ponderation)
data_ponderation['dates'] = pd.to_datetime(data_ponderation['dates'], format = "%Y-%m-%d", errors = "ignore")

# Calcul de la qualité de l'image pour la jussie et la crassule
list_valeurs_jussie = []

for index1, row1 in data_ponderation.iterrows():
    for index2, row2 in df_basepd_jussie.iterrows():
        if (row1['dates']==pd.Timestamp(row2['dates'])):
            list_valeurs_jussie.append(row2['valeurs'])

abondance = 0
for valeur in range(len(list_valeurs_jussie)):
    abondance = abondance + (((list_valeurs_jussie[valeur])/1000)*(data_ponderation['abondance'][valeur]))/sum(list_valeurs_jussie)
    
qj = abondance
qj = float("{0:.2f}".format(qj))

print(f"La classification par démélengeage a une qualité de {qj*100}% en {annee}", file=open(results_txt, "a"))


# ## Comparaison multitemporelle pour la jussie

# In[36]:


print("comparaison multi date")

de = f"path/Download{annee}/images/ete_conserve/"

## pour un même site va récuperer les valeurs max issues du démélange (recouvrement max de jussie pour un pixel)

liste_ete=[]

for img in ds:
    if img.endswith("fcls_jussie.tif"):
        print(img)
        with rasterio.open(f"{base}/{img}") as src_data:
            data=src_data.read()
        
        with rasterio.open(f"{de}/{img[0:19]}/{img[0:19]}_B11_clipped.tif") as src_crassule:
            data_crassule = src_crassule.read()
            data_crassule = np.where((data_crassule > 1500),1,0)
        
        data_jussie = data*data_crassule
        
        liste_ete.append(data_jussie)

        # Ajout à la pondération
        amax = np.amax(data_jussie, axis = 0)

st=np.stack(liste_ete)
print(st.shape)

#recupere les pourcentages max de recouvrement toute date confondues
st=np.amax(st,axis=0)
st=st[0,:,:]

with rasterio.open(f"{base}/{ds[0]}") as src:
    out_meta = src.meta.copy()
    with rasterio.open(f"{dossier_final}/demel_jussie_{annee}0701.tif", "w", **out_meta) as dest:
        dest.write(st,1)


# ## Comparaison multitemporelle pour la crassule

# In[37]:


## pour un même site va récuperer les valeurs max issues du démélange (recouvrement max de jussie pour un pixel)

liste_ete=[]

for img in ds:
    if img.endswith("fcls_crassule.tif"):
        print(img)
        with rasterio.open(f"{base}/{img}") as src_data:
            data=src_data.read()
        
        with rasterio.open(f"{de}/{img[0:19]}/{img[0:19]}_B8_clipped.tif") as src_jussie:
            data_jussie = src_jussie.read()
            data_jussie = np.where((data_jussie >= 4000),0,1)
        
        data_crassule = data*data_jussie
        
        liste_ete.append(data_crassule)

        # Ajout à la pondération
        amax = np.amax(data_crassule, axis = 0)

st=np.stack(liste_ete)
print(st.shape)

#recupere les pourcentages max de recouvrement toute date confondues
st=np.amax(st,axis=0)
st=st[0,:,:]

with rasterio.open(f"{base}/{ds[0]}") as src:
    out_meta = src.meta.copy()
    with rasterio.open(f"{dossier_final}/demel_crassule_{annee}0701.tif", "w", **out_meta) as dest:
        dest.write(st,1)


# ## Création des masques pour les couches d'alertes

# In[42]:


## Détermination du dossier d'alertes
dossier_alertes = f"path/Inventaire_invasive_{annee}/"

if os.path.exists(dossier_alertes):
    msk_jussie = f"{dossier_alertes}/msk_jussie.tif"
    msk_crassule = f"{dossier_alertes}/msk_crassule.tif"

## Si l'inventaire des plantes invasives de l'année en cours n'a pas été déposé sur le serveur, on prend
## le masque en format raster de l'année 2021 (année la plus récente)
if not os.path.exists(dossier_alertes):
    dossier_alertes = f"path/Inventaire_invasive_2021/"
    msk_jussie = f"{dossier_alertes}/msk_jussie.tif"
    msk_crassule = f"{dossier_alertes}/msk_crassule.tif"


# ## Couches d'alertes

# In[45]:


# Pour toutes les images de démélengeage
## Ouverture image de démélengeage
with rasterio.open(f"{dossier_final}/demel_{annee}0701.tif") as src:
    fcls = src.read()

## Ouverture masque de la jussie    
with rasterio.open(msk_jussie) as src_msk_j:
    msk_j = src_msk_j.read()
    msk_j[msk_j != -9999]=1
    msk_j[msk_j == -9999]=0

## Ouverture masque de la crassule
with rasterio.open(msk_crassule) as src_msk_c:
    msk_c = src_msk_c.read()
    msk_c[msk_c != -9999]=1
    msk_c[msk_c == -9999]=0
    
fcls_msk = fcls*msk_j*msk_c

## Sauvegarde
out_meta = src.meta.copy()   
out_meta.update({'nodata':0})

with rasterio.open(f"{dossier_final}/alertes_{annee}0701.tif", "w", **out_meta) as dst:
    dst.write(fcls_msk)
    
fcls_msk = fcls = msk_j = msk_c = None

##############################################################################
##############################################################################

# Pour la jussie
## Ouverture image de démélengeage
with rasterio.open(f"{dossier_final}/demel_jussie_{annee}0701.tif") as src:
    fcls = src.read()

## Ouverture masque de la jussie    
with rasterio.open(msk_jussie) as src_msk_j:
    msk_j = src_msk_j.read()
    msk_j[msk_j != -9999]=1
    msk_j[msk_j == -9999]=0
    
fcls_msk = fcls*msk_j

out_meta = src.meta.copy()   
out_meta.update({'nodata':0})

with rasterio.open(f"{dossier_final}/alertes_jussie_{annee}0701.tif", "w", **out_meta) as dst:
    dst.write(fcls_msk)

##############################################################################
##############################################################################
    
# Pour la crassule
## Ouverture image de démélengeage
with rasterio.open(f"{dossier_final}/demel_crassule_{annee}0701.tif") as src:
    fcls = src.read()
    
## Ouverture masque de la crassule
with rasterio.open(msk_crassule) as src_msk_c:
    msk_c = src_msk_c.read()
    msk_c[msk_c != -9999]=1
    msk_c[msk_c == -9999]=0
    
fcls_msk = fcls*msk_c

## Sauvegarde
out_meta = src.meta.copy()   
out_meta.update({'nodata':0})

with rasterio.open(f"{dossier_final}/alertes_crassule_{annee}0701.tif", "w", **out_meta) as dst:
    dst.write(fcls_msk)

fcls_msk = fcls = msk_j = msk_c = None


# ## Tuilage et droits

# In[50]:


# Application du tuilage et de l'overview

## Pour les images de démélengeage
script = "path/geotiff/tiff2overview.py"
ds_demel = "path/demelengeage/"
ds_demelj = "path/demel_jussie/"
ds_demelc = "path/demel_crassule/"

cmd = f'python {script} -i {dossier_final}/demel_{annee}0701.tif -n 20 -d {ds_demel} -t 512 -c NONE'
os.system(cmd)
cmd = f'python {script} -i {dossier_final}/demel_jussie_{annee}0701.tif -n 30 -d {ds_demelj} -t 512 -c NONE'
os.system(cmd)
cmd = f'python {script} -i {dossier_final}/demel_crassule_{annee}0701.tif -n 30 -d {ds_demelc} -t 512 -c NONE'
os.system(cmd)

## Pour les images d'alertes
ds_alertes = "path/alertes/"
ds_alertesj = "path/alertes_jussie/"
ds_alertesc = "path/alertes_crassule/"

cmd = f'python {script} -i {dossier_final}/alertes_{annee}0701.tif -n 20 -d {ds_alertes} -t 512 -c NONE'
os.system(cmd)
cmd = f'python {script} -i {dossier_final}/alertes_jussie_{annee}0701.tif -n 30 -d {ds_alertesj} -t 512 -c NONE'
os.system(cmd)
cmd = f'python {script} -i {dossier_final}/alertes_crassule_{annee}0701.tif -n 30 -d {ds_alertesc} -t 512 -c NONE'
os.system(cmd)

# Suppression du dossier et des images
shutil.rmtree(f"path{annee}/")

# Droit d'accès linux
cmd = f'chmod -R 777 path/download/'
os.system(cmd)

