#Author: Van Eylen code modified by Prajwal Niraula
#Institution: Wesleyan University
#Last Updated: March 1, 2017

from __future__ import division
from run_pipeline import run
from glob import glob
import re
from ExtractFlux import PredeterminedAperture
import warnings
warnings.filterwarnings("ignore") #To suppress the warning. Comment this to see the range of warning.

outputpath = 'PCurves/'
SubFolder = "PhaseCurves"



inputpath = "/home/pniraula/Downloads/PhaseCurveFitsFiles/*.fits"
#inputpath = "/home/pniraula/Downloads/*.fits"

filepaths = glob(inputpath)


i = 0
exc_list = []


for filepath in filepaths:
  EPIC_ID = re.search('[0-9]{9}',filepath).group(0)
  print "Currently running EPIC ID::", EPIC_ID

  try:
    run(filepath=filepath,outputpath=outputpath,CNum=None,chunksize=100,method ='SFF',SubFolder=SubFolder)

  except Exception as inst:
    print inst
    exc_list.append(inst)

  i = i + 1
if exc_list:
  print 'Module failed for some stars:'
  print exc_list

print 'Done...'
