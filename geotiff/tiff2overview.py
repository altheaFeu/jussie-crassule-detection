#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from optparse import OptionParser

# Paramètres en entrée du script
parser = OptionParser()
parser.add_option("-c", "--compress", dest="compress", help="Métode de compression",default='DEFLATE')
parser.add_option("-n", "--nprefix", dest="nprefix", type="int", help="Nombre de caractères du préfixe conservés",default=24)
parser.add_option("-i", "--input", dest="inputtif", help="Geotiff en entree",default='')
parser.add_option("-d", "--dir", dest="outputdir", help="Dossier destination",default='/tmp')
parser.add_option("-t", "--tilesize", dest="tilesize", help="Taille de la tuile",default="512")
(options, args) = parser.parse_args()

# Formatage d'image de sortie
outputtif = os.path.basename(options.inputtif)
outputtif = options.outputdir+"/"+outputtif [0:options.nprefix]
print ("Image de sortie : "+outputtif)
# tuilage de l'image source
cmd = "gdal_translate -co COMPRESS=" + options.compress + " -co 'TILED=YES' -co 'BLOCKXSIZE=" + options.tilesize + "' -co 'BLOCKYSIZE=" + options.tilesize + "' " + options.inputtif + " " + outputtif
#print (cmd)
os.system(cmd)

# Ajout de l'overview
cmd = "gdaladdo -r cubic " + outputtif + " 2 4 8 16 32 64 128"
#print (cmd)
os.system(cmd)


