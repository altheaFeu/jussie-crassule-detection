#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Besoin d'installer en premier gsconfig-py3
from geoserver.catalog import Catalog
from glob import glob
import xml.etree.ElementTree as ET 
import requests
import pandas as pd
import urllib3
import warnings 

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

## année
annee = str(datetime.date.today().year)


# ## Modification des images mosaïques

# In[100]:


# Connexion au geoserver
cat = Catalog("link-to-catalog",
             username = "username", password="password")

Dossier = "path-do-files"

# SVM
svm = cat.get_store("SVM")
cat.harvest_externalgranule(f"{Dossier}/SVM/SVM_{annee}0701.tif", svm)


############################################################
# Alertes démélengeage
alertes = cat.get_store("alertes")
cat.harvest_externalgranule(f"{Dossier}/alertes/alertes_{annee}0701.tif", alertes)

alertes_j = cat.get_store("alertes_jussie")
cat.harvest_externalgranule(f"{Dossier}/alertes_jussie/alertes_jussie_{annee}0701.tif", 
                            alertes_j)

alertes_c = cat.get_store("alertes_crassule")
cat.harvest_externalgranule(f"{Dossier}/alertes_crassule/alertes_crassule_{annee}0701.tif", 
                            alertes_c)

############################################################
# Démélengeage
demel = cat.get_store("demel")
cat.harvest_externalgranule(f"{Dossier}/demelengeage/demel_{annee}0701.tif", 
                            demel)

demel_j = cat.get_store("demel_jussie")
cat.harvest_externalgranule(f"{Dossier}/demel_jussie/demel_jussie_{annee}0701.tif", 
                            demel_j)

demel_c = cat.get_store("demel_crassule")
cat.harvest_externalgranule(f"{Dossier}/demel_crassule/demel_crassule_{annee}0701.tif", 
                            demel_c)

############################################################
# Zones inondées et plans d'eau
inondees = cat.get_store("zones_inondees")
cat.harvest_externalgranule(f"{Dossier}/zones_inondees/SENTINEL2A_{annee}0101.tif", 
                            inondees)

plan_eau = cat.get_store("plan_eau")
cat.harvest_externalgranule(f"{Dossier}/plan_eau/plan_eau_{annee}0201.tif", 
                            plan_eau)


# ## Modification du fichier xml

# In[98]:


# Connexion au drive de l'application et récupération du fichier de configuration
response = requests.get("https://geosas.fr/drive/remote.php/webdav/apps_pnr/config.xml",
                        auth=('afeuillet', 'Althea35!'), verify = False)

# Enregistrement du fichier de configuration en local
with open('/home/afeuillet/Documents/Data/config.xml', "w") as file:
    file.write(response.text)

# Ouverture du fichier xml
tree = ET.parse('oath/config.xml')
root = tree.getroot()

## Réécriture du fichier xml avec les nouvelles valeurs dans "timevalues"
tree.write('oath/config.xml')

# Enregistrement du nouveau fichier de configuration sur le serveur
with open('oath/config.xml') as f :
    response = requests.put("link-to-server/config.xml", 
                             auth=('username', 'password'), data = f)

