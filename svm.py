#!/usr/bin/env python
# coding: utf-8

# ### Packages

# In[76]:


#!/usr/bin/env python

import rasterio
from rasterio.mask import mask
from rasterio.plot import show
from rasterio.plot import show_hist
from rasterio.windows import Window
from joblib import Parallel, delayed
import joblib
from rasterio.plot import reshape_as_raster, reshape_as_image
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import mapping
from osgeo import gdal, gdal_array
import matplotlib.pyplot as plt
import os, sys
from glob import glob
import shutil
import datetime
import gdal
import time

from sklearn.svm import SVC
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import seaborn as sns

annee = str(datetime.date.today().year)


# ### Recherche des nuages

# In[59]:


def nuage(dossier, chemin):
    # Importation de l'image des nuages
    with rasterio.open(f"{chemin}/{dossier}/{dossier}_CLM_R1_clipped.tif") as src_nuage:
        data_nuage=src_nuage.read(1)
        
        # Création d'un pandas dataframe pour connaître le nombre de pixel pour chaque valeur
        unique, count = np.unique(data_nuage, return_counts = True)
        raster_dataframe = pd.DataFrame()
        raster_dataframe['Pixel Value']=unique
        raster_dataframe['Count']=count
        
        # Calcul du nombre total de pixel
        sum_total = raster_dataframe['Count'].sum()

        count = 0

        # Comptage du nombre de pixels égal à 0 et 1
        if 0 in raster_dataframe['Pixel Value'].values:
            count = count + float(raster_dataframe.loc[raster_dataframe['Pixel Value']==0, 'Count']) 

        if 2 in raster_dataframe['Pixel Value'].values:
            count = float(raster_dataframe.loc[raster_dataframe['Pixel Value']==2, 'Count'])

        # Calcul du pourcentage de nuage
        pourcentage = 100-((count/sum_total)*100)

        pourcentage = float(pourcentage)
        pourcentage = round(pourcentage, 2)
        
        print("\n{}\n".format(dossier))
        print(f"Il y a {pourcentage}% de nuages\n")

        return pourcentage


# ### Importation des données

# In[60]:


# Recherche de toutes les bandes sentinel dans le dossier
chemin = "path/images/ete_conserve/"

sentinel_band_paths = []
band_list = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
nb = 0

for folder in os.listdir(chemin):
    images_dossier = []
    pourcentage = nuage(folder, chemin)
    
    # On ne sélectionne que les images avec 0% de nuage
    if pourcentage <= 1:
        for i, band in enumerate(band_list):
            # Recherche des bandes dans l'ordre pour pouvoir les stacker
            for file in glob(f"{chemin}/{folder}/{folder}_{band}_clipped.tif") :
                images_dossier.append(file)

        images_dossier = [img for i, img in enumerate(images_dossier)]
        
        # Ajout du nouveau dossier
        sentinel_band_paths.extend(images_dossier)
        
        print("\n{}: \n{}".format(folder, images_dossier))
        print("\n{}: \n{}".format(folder, sentinel_band_paths))

        nb += 10
            
print(f"Nb bandes : {nb}")


# ### Réalisation d'un stack

# In[68]:


# Modification des métadonnées du stack
with rasterio.open(sentinel_band_paths[0]) as src0:
    meta = src0.meta
    meta.update({'nodata':-9999,
                'count':nb})

dossier_resultat = f"path{annee}/SVM"

if not os.path.exists(dossier_resultat):
    os.makedirs(dossier_resultat)
    
    
print ("             ### Début du stackage ###             ") 
start = time.time()

# Stack de toutes les bandes
with rasterio.open(f'{dossier_resultat}/stack_final_SVM.tif', 'w', **meta) as dst:
    for id, layer in enumerate(sentinel_band_paths, start = 1):
        print(layer, id)
        # Ecriture de la bande avec son numéro
        with rasterio.open(layer) as src1:
            dst.write_band(id, src1.read(1))
        
        print("\n")
                
print ("             ### Stackage total terminé ###             ") 
diff_time = time.time() - start
minutes = diff_time // 60
seconds = diff_time % 60
print(f"Images démélengées en {minutes}min {seconds}s\n")


# # Classification SVM

# In[62]:


dossier_resultat = f"path{annee}/SVM"
if not os.path.exists(dossier_resultat):
    os.makedirs(dossier_resultat)

dossier_sortie = "path/SVM/"
results_txt = f'{dossier_sortie}/infos_sup/SVM_{annee}.txt'

# Ouverture du stack finale et récupération des informations sur l'image
ds = gdal.Open(f'{dossier_resultat}/stack_final_SVM.tif')
rows, cols = ds.RasterXSize, ds.RasterYSize
bands = ds.RasterCount
dsType = gdal_array.GDALTypeCodeToNumericTypeCode(ds.GetRasterBand(1).DataType)
dsProj = ds.GetProjection()
dsTransform = ds.GetGeoTransform()

print("Shape: %s | type: %s | Bandes: %s" % ((rows, cols), dsType, bands))

imgArr = np.zeros((cols, rows, bands), dsType)
for band in range(bands):
    imgArr[:,:,band] = ds.GetRasterBand(band+1).ReadAsArray()


# ## Données d'entraînement

# In[63]:


training = "path/train-data.shp"

# Ouverture des données d'entraînement en format raster
with rasterio.open(f'{dossier_resultat}/stack_final_SVM.tif') as src:
    err = gdal.Rasterize('', training,
                         outputBounds = [src.bounds[0],src.bounds[1],src.bounds[2],src.bounds[3]], 
                         attribute = 'classe',
                         format = 'MEM', xRes = 10, yRes = 10)
roi = err.ReadAsArray()


# ## Préparation des données

# In[64]:


# X = découpage de l'image avec l'image d'entraînement
# Y = Masque de tout les classes sur l'image d'entraînement

trainX = imgArr[roi > 0]
trainY = roi[roi>0]
print("Shape pour X :", trainX.shape)
print("Shape pour Y :", trainY.shape)

newshape = (imgArr.shape[0]*imgArr.shape[1], imgArr.shape[2])
testArr = imgArr.reshape(newshape)
print(f"{imgArr.shape} est changé en {testArr.shape}")


# In[69]:


start = time.time()

# Entraînement du modèle et test de différents paramètres
print("Paramètres SVM")
param_grid = {'kernel': ['linear'], "gamma": [1e-3, 1e-4], "C": [1, 10]}

# Sélection des meilleurs paramètres pour lancer le SVM
svc = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
svc.fit(trainX, trainY)

# Lancement de la classification
predictions = svc.predict(testArr)

print("          ### Fin de la classification ###          ")
diff_time = time.time() - start
minutes = diff_time//60
secondes = diff_time%60
print(f"Classification réalisée en {minutes} min {secondes} sec\n")


# In[25]:


## Tableau de validation croisée
df = pd.DataFrame()
df['truth'] = trainY
df['predict'] = svc.predict(trainX)
print(pd.crosstab(df['truth'], df['predict'], margins=True))
print(pd.crosstab(df['truth'], df['predict'], margins=True), file=open(results_txt, "a"))
print(accuracy_score(trainY, svc.predict(trainX))*100, file=open(results_txt, "a"))


# In[26]:


# Nom et dossier de la matrice de confusion
exp_im = f'Matrice de confusion de la classification \n SVM pour l\'annee {annee}'
cm_directory2 = f'path/SVM_{annee}_matrix_validation.png'

# Matrice de confusion

## Styles graphiques
Title = {'fontname':'Calibri', 'fontsize': '20'}
Axis = {'fontname':'Calibri', 'fontsize': '13'}

## Exportation de la matrice de confusion
cm = confusion_matrix(trainY, svc.predict(trainX))
plt.figure(figsize=(10,7))
cm = sns.heatmap(cm, square=True, cmap='coolwarm',linewidths=.7, annot=True, fmt='d', )
plt.title('Model - Confusion matrix ' + exp_im, Title, pad=20)
plt.xlabel('classes - predicted', Axis)
plt.ylabel('classes - truth', Axis)
plt.yticks(np.arange(4)+0.5,('Ludwigia g.','Crassula h.','Water','Other'), fontname='Calibri', fontsize="9")
plt.xticks(np.arange(4)+0.5,('Ludwigia g.','Crassula h.','Water','Other'), fontname='Calibri', fontsize="9")
plt.savefig(cm_directory2)
print('Model - Confusion matrix saved in {}'.format(cm_directory2), file=open(results_txt, "a"))
print('-------------------------------------------------', file=open(results_txt, "a"))


# ## Sauvegarde de la classification

# In[74]:


# Sauvegarde de la classification
def save_classified(data, outfile):
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()
    
    outDataset = driver.Create(outfile, rows, cols, 1, gdal.GDT_Byte)
    outDataset.SetGeoTransform(dsTransform)
    outDataset.SetProjection(dsProj)
    outBand = outDataset.GetRasterBand(1)
    outBand.WriteArray(data, 0, 0)

imgShp = imgArr.shape[:-1]

save_classified(predictions.reshape(imgShp), f"{dossier_resultat}/SVM_{annee}_nomsk.tif")

# Application des masques sur l'image SVM
## Ouverture de l'image sans masque
with rasterio.open(f"{dossier_resultat}/SVM_{annee}_nomsk.tif") as src_data :
    data = src_data.read()
    data[data==-9999]=0

## Ouverture du masque des zones salées
with rasterio.open("path/msk_sel.tif") as src_sel :
    data_sel = src_sel.read()
    data_sel[data_sel==-9999]=0

## Ouverture du masque des canalisations
with rasterio.open("path/msk_canalisations.tif") as src_canal :
    data_canal = src_canal.read()
    data_canal[data_canal==-9999]=0

## Ouverture du masque des villes et la BD topage
with rasterio.open("path/msk_ville_topage.tif") as src_ville :
    data_ville = src_ville.read()
    data_ville[data_ville==-9999]=0
    
## Ouverture du masque des prairies
for prairies in glob(f"path/Download{annee}/msk_prairies/**.tif"):
    print(prairies)
    with rasterio.open(prairies) as src_prairies :
        data_prairies = src_prairies.read()
        data_prairies[data_prairies==-9999]=0

## Ouverture ud masque des bocages
with rasterio.open("/home/afeuillet/Documents/Data/msk/msk_bocages2.tif") as src_boc :
    data_boc = src_boc.read()
    data_boc[data_boc!=-9999]=1
    data_boc[data_boc==-9999]=0
    print(np.min(data_boc))
    print(np.max(data_boc))

data_final = data*data_boc*data_canal*data_sel*data_ville*data_prairies

out_meta = src_data.meta.copy()

with rasterio.open(f"{dossier_resultat}/SVM_{annee}0701.tif", "w", **out_meta) as dest:
    dest.write(data_final)

os.remove(f"{dossier_resultat}/SVM_{annee}_nomsk.tif")


# ## Validation

# In[34]:


print('-------------------------------------------------', file=open(results_txt, "a"))
print('VALIDATION', file=open(results_txt, "a"))

validation = "path/valid_data.shp"

## Ouverture des données de validation en format raster
with rasterio.open(f'{dossier_resultat}/stack_final_SVM.tif') as src:
    err_v = gdal.Rasterize('', validation,
                         outputBounds = [src.bounds[0],src.bounds[1],src.bounds[2],src.bounds[3]], attribute = 'classe',
                         format = 'MEM', xRes = 10, yRes = 10)
roi_v = err_v.ReadAsArray()


# In[35]:


# Recherche des valeurs différentes de 0
n_val = (roi_v > 0).sum()
print('{n} validation pixels'.format(n=n_val))
print('{n} validation pixels'.format(n=n_val), file=open(results_txt, "a"))

# Recherche des labels de la validation
labels_v = np.unique(roi_v[roi_v > 0])
print('validation data include {n} classes: {classes}'.format(n=labels_v.size, classes=labels_v))
print('validation data include {n} classes: {classes}'.format(n=labels_v.size, classes=labels_v), file=open(results_txt, "a"))

# X = découpage de l'image des prédictions avec l'image de validation
# Y = Masque de tout les classes sur l'image d'entraînement
imgShp = imgArr.shape[:-1]
reshaped_predictions = predictions.reshape(imgShp)

X_v = reshaped_predictions[roi_v > 0] 
y_v = roi_v[roi_v > 0]

print('Our X matrix is sized: {sz_v}'.format(sz_v=X_v.shape))
print('Our y array is sized: {sz_v}'.format(sz_v=y_v.shape))
print('-------------------------------------------------', file=open(results_txt, "a"))

# Validation croisée
convolution_mat = pd.crosstab(y_v, X_v, margins=True)

#convolution_mat = confusion_matrix(y_v, X_v)
print(convolution_mat)
print(convolution_mat, file=open(results_txt, "a"))
print('-------------------------------------------------', file=open(results_txt, "a"))

target_names = list()
for name in range(1,(labels_v.size)+1):
    target_names.append(str(name))
sum_mat = classification_report(y_v,X_v,target_names=target_names)
print(sum_mat)
print(sum_mat, file=open(results_txt, "a"))
print('-------------------------------------------------', file=open(results_txt, "a"))

# Calcul de l'Overall Accuracy Score (OAA)
print('OAA = {} %'.format(accuracy_score(y_v,X_v)*100))
print('OAA = {} %'.format(accuracy_score(y_v,X_v)*100), file=open(results_txt, "a"))
print('-------------------------------------------------', file=open(results_txt, "a"))


# In[37]:


cm_directory = f'path/SVM_{annee}_matrix_calibration.png'

# Style graphique
Title = {'fontname':'Calibri', 'fontsize': '20'}
Axis = {'fontname':'Calibri', 'fontsize': '13'}

# Exportation de la matrice de confusion - validation
cm_val = confusion_matrix(roi_v[roi_v > 0], reshaped_predictions[roi_v > 0])
plt.figure(figsize=(10,7))
cm_val = sns.heatmap(cm_val, square=True, cmap='coolwarm',linewidths=.7, annot=True, fmt='d', )
plt.title(exp_im, Title, pad=20)
plt.xlabel('classes - predicted', Axis)
plt.ylabel('classes - truth', Axis)
plt.yticks(np.arange(4)+0.5,('Ludwigia g.','Crassula h.','Water','Other'), fontname='Calibri', fontsize="9")
plt.xticks(np.arange(4)+0.5,('Ludwigia g.','Crassula h.','Water','Other'), fontname='Calibri', fontsize="9")
plt.savefig(cm_directory)
print('Validation - Confusion matrix saved in {}'.format(cm_directory), file=open(results_txt, "a"))
print('-------------------------------------------------', file=open(results_txt, "a"))


# ## Tuilage et droits

# In[75]:


# Application du tuilage et de l'overview

script = "path/geotiff/tiff2overview.py"
ds_svm = "path/download/SVM/"

# Pour le NDVI en été
cmd = f'python {script} -i {dossier_resultat}/SVM_{annee}0701.tif -n 23 -d {ds_svm} -t 512 -c NONE'
os.system(cmd)

# Suppression du dossier et des images
shutil.rmtree(dossier_resultat)

# Droit d'accès linux
cmd = f'chmod -R 777 {ds_svm}'
os.system(cmd)

