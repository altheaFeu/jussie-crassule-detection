#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
from optparse import OptionParser

# Paramètres en entrée du script
parser = OptionParser()
parser.add_option("-c", "--compress", dest="compress", help="Méthode de compression",default='DEFLATE')
parser.add_option("-n", "--nprefix", dest="nprefix", type="int", help="Nombre de caractères du préfixe conservés",default=24)
parser.add_option("-i", "--input", dest="inputdir", help="Dossier contenant les geotiff à traiter",default='')
parser.add_option("-o", "--dir", dest="outputdir", help="Dossier destination",default='/tmp')
parser.add_option("-p", "--pattern", dest="pattern", help="Pattern pour sélectionner/filtrer les images",default='*')
parser.add_option("-t", "--tilesize", dest="tilesize", help="Taille de la tuile",default="512")
(options, args) = parser.parse_args()


inputtif_list = glob.glob(options.inputdir + '/*'+options.pattern + '*' )
print(inputtif_list)

for inputtif in inputtif_list:
    print(inputtif)

# Formatage d'image de sortie
    outputtif = os.path.basename(inputtif)
    outputtif = options.outputdir+"/"+outputtif [0:options.nprefix]+".tif"
    print ("Image de sortie : "+outputtif)

# tuilage de l'image source
    cmd = "gdal_translate -co COMPRESS=" + options.compress + " -co 'TILED=YES' -co 'BLOCKXSIZE=" + options.tilesize + "' -co 'BLOCKYSIZE=" + options.tilesize + "' " + inputtif + " " + outputtif
    os.system(cmd)

# Ajout de l'overview
    cmd = "gdaladdo -r cubic " + outputtif + " 2 4 8 16 32 64 128"
    os.system(cmd)


