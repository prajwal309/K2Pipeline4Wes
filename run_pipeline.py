'''
% Main pipeline file to generate K2 photometry starting from .fits pixel files downloaded from K2 MAST
% Author Vincent Van Eylen
% Modified by Prajwal Niraula
% Contact vincent@phys.au.dk
% See Van Eylen et al. 2015 (ApJ) for details. Please reference this work if you found this code helpful!
'''
import matplotlib
matplotlib.use('Agg')

# general python files
import os
import matplotlib.pyplot as pl
import numpy as np
import time

# pipeline files
import pixeltoflux
import centroidfit
import periodfinder
import re
import K2SFF

from ExtractFlux import GetLightCurve

def run(filepath='',outputpath='',chunksize=300,method ='SFF',SubFolder=False):

  # Takes strings with the EPIC number of the star and input/outputpath. Campaign number is used to complete the correct filename as downloaded from MAST
  starname = re.search('[0-9]{9}',filepath).group(0)

  #handling case for spd vs lpd
  if "spd" in filepath:
    starname = starname+"_spd"

  outputfolder = os.path.join(outputpath,str(starname))

  #extract campaign number from filename
  CmpStr = re.search('c[0-9]{2}',filepath).group(0)
  CNum = int(CmpStr[1:])


  if CNum>8 and CNum<12:

        #two filenames present for campaign 9 and campaign 10
        AdditionalFilepath = filepath.replace("1_","2_")

        t1,f_t1,Xc1,Yc1 = PredeterminedAperture(filepath,outputpath=outputpath,plot=False,SubFolder=SubFolder)
        #t1,f_t1,Xc1,Yc1 = [np.array([]),np.array([]),np.array([]),np.array([])]
        t2,f_t2,Xc2,Yc2 = PredeterminedAperture(AdditionalFilepath,outputpath=outputpath,plot=False,SubFolder=SubFolder)



        T_Raw = np.concatenate((t1,t2), axis=0)
        Flux_Raw = np.concatenate((f_t1,f_t2), axis=0)

        #t1,f_t1,Xc1,Yc1 = centroidfit.find_thruster_events(t1,f_t1,Xc1,Yc1,starname=starname,outputpath=outputfolder)
        #t2,f_t2,Xc2,Yc2 = centroidfit.find_thruster_events(t2,f_t2,Xc2,Yc2,starname=starname,outputpath=outputfolder)

  else:
        t,f_t,Xc,Yc = GetLightCurve(filepath,outputpath=outputpath,plot=False)
        T_Raw = np.copy(t)
        Flux_Raw = np.copy(f_t)

        #Remove the thruster events #TODO uncomment this later or implement this in K2SFF
        #t,f_t,Xc,Yc = centroidfit.find_thruster_events(t,f_t,Xc,Yc,starname=starname,outputpath=outputfolder)

  # now fit a polynomial to the data (inspired by Spitzer data reduction), ignore first data points which are not usually very high-quality
  if method == 'Spitzer':
        print ("Running Spitzer")
        if CNum>8 and CNum<12:
            [t1,f_t1] = centroidfit.spitzer_fit(t1,f_t1,Xc1,Yc1,starname=starname,outputpath=outputpath,chunksize=chunksize)
            [t2,f_t2] = centroidfit.spitzer_fit(t2,f_t2,Xc2,Yc2,starname=starname,outputpath=outputpath,chunksize=chunksize)
            t = np.append(t1,t2)
            f_t = np.append(f_t1,f_t2)
            del t1, t2, f_t1, f_t2
        #elif CNum==1:
            #[t,f_t] = centroidfit.spitzer_fit(t[90:],f_t[90:],Xc[90:],Yc[90:],starname=starname,outputpath=outputpath,chunksize=chunksize)
        else:
            [t,f_t] = centroidfit.spitzer_fit(t,f_t,Xc,Yc,starname=starname,outputpath=outputpath,chunksize=chunksize)

  elif method == 'SFF':
        print ("Running SFF")
        if CNum>8 and CNum<12:
            #[t1,f_t1] = centroidfit.sff_fit(t1,f_t1,Xc1,Yc1,starname=starname,outputpath=outputpath,chunksize=chunksize)
            #[t2,f_t2] = centroidfit.sff_fit(t2,f_t2,Xc2,Yc2,starname=starname,outputpath=outputpath,chunksize=chunksize)

            [t1,f_t1] = K2SFF.K2SFF_VJ14(t1,f_t1,Xc1,Yc1,CmpStr, starname=starname,outputpath=outputpath,chunksize=chunksize)
            [t2,f_t2] = K2SFF.K2SFF_VJ14(t2,f_t2,Xc2,Yc2,CmpStr, starname=starname,outputpath=outputpath,chunksize=chunksize)
            t = np.append(t1,t2)
            f_t = np.append(f_t1,f_t2)
        #elif CNum==1:
            #[t,f_t] = centroidfit.sff_fit(t[90:],f_t[90:],Xc[90:],Yc[90:],starname=starname,outputpath=outputpath,chunksize=chunksize)
        else:
            #Girish method
            #[t,f_t] = centroidfit.sff_fit(t,f_t,Xc,Yc,starname=starname,outputpath=outputpath,chunksize=chunksize)

            #My method
            [t,f_t] = K2SFF.K2SFF_VJ14(t,f_t,Xc,Yc,CmpStr, starname=starname,outputpath=outputpath,chunksize=chunksize)
  else:
        raise Exception('No valid method given.')


  T_Detrended = np.copy(t)
  Flux_Detrended = np.copy(f_t)
  #[t,f_t] = centroidfit.clean_data(t,f_t) # do a bit of cleaning
  np.savetxt(os.path.join(outputfolder, 'CleanedLightCurve.csv'),np.transpose([t,f_t]),header='Time, Flux', delimiter=',')

  T_Cleaned = np.copy(t)
  Flux_Cleaned = np.copy(f_t)

  pl.figure(figsize=(15,10))
  pl.subplot(3,1,1)
  pl.plot(T_Raw, Flux_Raw, "ko", MarkerSize=2)
  pl.ylabel("Flux Counts")
  pl.title("Raw Flux")

  pl.subplot(3,1,2)
  pl.plot(T_Detrended, Flux_Detrended, "ko", MarkerSize=2)
  pl.ylabel("Flux Counts")
  pl.title("Detrended Flux")

  pl.subplot(3,1,3)
  pl.plot(T_Cleaned, Flux_Cleaned, "ko", MarkerSize=2)
  pl.xlabel("Time (days)")
  pl.ylabel("Flux Counts")
  pl.title("Cleaned Flux")

  pl.suptitle(starname)
  pl.savefig(outputfolder+"/DiagnosticPlot.png")
  pl.close('all')

  del T_Raw, T_Detrended, T_Cleaned, Flux_Raw, Flux_Detrended, Flux_Cleaned


  folded_date,f_t_folded,period,freqlist,powers = periodfinder.get_period(t,f_t,outputpath=outputpath,starname=starname)
  periodfinder.make_combo_figure(filepath, t,f_t,period,freqlist,powers,starname= starname,outputpath=outputpath)
  pl.close('all')
