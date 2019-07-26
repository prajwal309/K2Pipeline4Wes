#Author: Van Eylen code modified by Prajwal Niraula
#Institution: Wesleyan University
#Last Updated: March 1, 2017


#Aperture system is given by the old van eylen method

from __future__ import division
import matplotlib
matplotlib.use('Qt4Agg')
#import multiprocessing as mp

import os


from run_pipeline import run
from glob import glob
import re

import warnings
warnings.filterwarnings("ignore") #To suppress the warning. Comment this to see the range of warning.

#inputpath = '/Volumes/westep/mikey/rerun_nopsf_star/*.fits'
inputpath = 'TestStars/*.fits'#'GO19051-targets/*.fits'
outputpath = 'UniquerTest' #GO19051-targets_Spitzer'
#outputpath = 'rerunstarsSpitzer'

#Create the folder if it does not exc_list
if not(os.path.exists(outputpath)):
    os.system('mkdir %s' %(outputpath))

filepaths = glob(inputpath)
print(filepaths)

allinput = [f.split('/')[-1].split('.')[0] for f in filepaths]
alloutput = [f.split('/')[-1] for f in glob(outputpath + '/*')]

'''
todo = []
for i in allinput:
  epicid = i[4:13]
  if epicid not in alloutput:
    todo.append('/Volumes/westep/ismael/Campaign18' + '/' + i + '.fits')
  else:
    epicid = i[4:13]
    print('%s already done' % epicid)
'''

for FILE in filepaths:#todo
  print (FILE)
  EPIC_ID = re.search('[0-9]{9}',FILE).group(0)
  print ("Now running::",EPIC_ID)
  fp = outputpath + '/' + str(EPIC_ID)

  chunksize = 150
  Campaign = re.search('c[0-9]{2}',FILE).group(0)
  #Campaign = 19#str(int(Campaign[1:]))
  run(filepath=FILE,outputpath=outputpath,chunksize=chunksize,method = 'Spitzer') # SFF or Spitzer


  '''
  if fp not in glob(outputpath + '/*'):
    try:
        chunksize = 150
        #Campaign = re.search('c[0-9]{2}',FILE).group(0)
        Campaign = 18#str(int(Campaign[1:]))
        run(filepath=FILE,outputpath=outputpath,chunksize=chunksize,method = 'Spitzer') # SFF or Spitzer
    except Exception as inst:
        print inst
  else:
    print '%s Already done' % EPIC_ID
  '''
print ('Completed the task')
