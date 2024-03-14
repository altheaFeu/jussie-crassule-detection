# Détection de la Jussie et de la crassule à partir d'images Sentinel 2 🌱

[![Construit par](https://img.shields.io/badge/Construit%20par-Althéa_Feuillet-orange.svg)](https://yourportfolio.com)
![Python](https://img.shields.io/badge/-Python-3776AB?style=flat&logo=python&logoColor=white)
![GeoServer](https://img.shields.io/badge/-GeoServer-BA686B?style=flat&logo=geoserver&logoColor=white)
![Shell](https://img.shields.io/badge/-Shell-4EAA25?style=flat&logo=gnu-bash&logoColor=white)

## Présentation
Ce script a été développé lors de mon stage à l'Agrocampus Ouest en 2022.

L'objectif est d'utiliser des images Sentinel-2 pour détecter la présence de la crassule et de la jussie dans le Parc Naturel de Brière.

Ce projet s'est déroulé en plusieurs étapes :

- Un script `lancement.sh` qui permet de lancer tous les processus.
- Un script `telechargement_seul` qui permet de télécharger les images pour une période donnée.
- Un script `svm.py` qui effectue une classification "Support Vector Machine".
- Un script `eau.py` qui calcule les masques d'eau.
- Un script `demelangeage_fcls.py` qui réalise un démêlage des données Sentinel.
- Un script `traitements_finaux.py` qui calcule les indices d'eau et de végétation (NDVI, NDWI, etc.) et applique les masques sur les résultats du démélangeage.
- Un script `geoserver.py` qui exporte et met à jour les données sur le serveur.

Ce travail a ensuite été automatisé à l'aide d'un crontab sur Linux.
