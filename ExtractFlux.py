from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.colors as colors
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import LSQUnivariateSpline as spline
from scipy.signal import gaussian
from scipy.ndimage import filters
from astropy.io import fits
from scipy.ndimage import convolve, measurements
import os
import re

import matplotlib as mpl
mpl.use('Tkagg')

import FindAperture

#this library is for downloading the image from previous surveys
from astroquery.skyview import SkyView

def RA_convert(RA):
  hour = (int(RA/360.0*24.0))
  minute = int((RA - hour*15.0)/15.0*60.0)
  second = ((RA - hour*15.0)/15.0*60.0 - int((RA - hour*15.0)/15.0*60.0))*60
  second = round(second,2)
  hour = str(hour)
  minute = str(minute)
  if len(hour)==1:
      hour = "0"+hour
  if len(minute)==1:
      minute="0"+minute
  if len(str(int(second)))==1:
      second = "0"+str(second)
  else:
      second = str(second)
  return hour+":"+minute+":"+second

def DEC_convert(DEC):
    if DEC<0:
        Sign="-"
    else:
        Sign = ""
    DEC=abs(DEC)
    Deg = int(DEC)
    minute = int((DEC - int(DEC))*60.0)
    second = (DEC - Deg - minute/60.0)*3600.0
    second = round(second,2)
    Deg = str(Deg)
    minute = str(minute)
    if len(Deg)==1:
        Deg = "0"+Deg
    if len(minute)==1:
        minute="0"+minute
    if len(str(int(second)))==1:
        second = "0"+str(second)
    else:
        second = str(second)
    return Sign+Deg+":"+minute+":"+second

def DownloadImage(RA, DEC, outputfolder):
    Pos = RA_convert(RA)+","+DEC_convert(DEC)
    print (RA, DEC)
    print (Pos)
    img = SkyView.get_images(position=Pos,survey=['DSS2 Blue','DSS2 IR','DSS2 Red'],pixels='50,50',coordinates='J2000',grid=True,gridlabels=True)
    #img = SkyView.get_images(position='22:57:00,62:38:00',survey=['DSS2 Blue','DSS2 IR','DSS2 Red'],pixels='100,100',coordinates='J2000',grid=True,gridlabels=True)

    fig,ax = pl.subplots(ncols=3,figsize=(24,8))
    plot = ax[0].imshow(img[0][0].data,vmax=np.max(img[0][0].data)*.65,vmin=np.max(img[2][0].data)*.45)
    plot1 = ax[1].imshow(img[1][0].data,vmax=np.max(img[1][0].data)*.75,vmin=np.max(img[1][0].data)*.4)
    plot2 = ax[2].imshow(img[2][0].data,vmax=np.max(img[2][0].data)*.485,vmin=np.max(img[2][0].data)*.25)
    pl.savefig(outputfolder+"/SkyImage.png")

from pixeltoflux import get_lightcurve

def ApertureOutline(StdAper,AvgFlux, outputfolder, X,Y):
    #find the outline and save the aperture in the relevant folder
    ver_seg = np.where(StdAper[:,1:] != StdAper[:,:-1])
    hor_seg = np.where(StdAper[1:,:] != StdAper[:-1,:])
    l = []

    for p in zip(*hor_seg):
        l.append((p[1], p[0]+1))
        l.append((p[1]+1, p[0]+1))
        l.append((np.nan,np.nan))

    for p in zip(*ver_seg):
        l.append((p[1]+1, p[0]))
        l.append((p[1]+1, p[0]+1))
        l.append((np.nan, np.nan))


    segments = np.array(l)
    pl.figure()
    pl.imshow(AvgFlux,cmap='gray',norm=colors.PowerNorm(gamma=1./2.),interpolation='none')
    pl.colorbar()
    pl.plot(segments[:,0]-0.5, segments[:,1]-0.5, color=(1,0,0,.5), linewidth=3)
    pl.plot(X,Y, "ro")
    pl.title("Aperture Selected")
    pl.gca().invert_yaxis()
    pl.axis('equal')
    pl.tight_layout()
    pl.savefig(outputfolder+"/Aperture.png")
    pl.close()


def RawFluxDiagram(Quality,TotalDate,FluxArray,outputfolder):
    IndexArray = [Quality==0]
    FluxArray = np.array(FluxArray)
    for i in range(21):
        IndexArray.append(Quality==(2**i))

    ColorList = ["black","orange","blue","green", "cyan", "magenta", "red"]
    MarkerList = ['o','^','*']
    pl.figure(figsize=(16,8))
    pl.clf()
    for i in range(22):
        pl.plot(TotalDate[IndexArray[i]], FluxArray[IndexArray[i]],label=str(i)+":"+str(np.sum(IndexArray[i])), MarkerSize=3, marker = MarkerList[i%3], color=ColorList[i%7], linestyle='none')
    pl.legend(loc='best')
    pl.savefig(outputfolder+'/QualityIndex.png')
    pl.close()

    QualityIndex = Quality==0
    NoQualityIndex = Quality!=0

    pl.figure(figsize=(15,5))
    pl.plot(TotalDate[QualityIndex], FluxArray[QualityIndex], "r*",MarkerSize=2,label="Qualified")
    pl.plot(TotalDate[NoQualityIndex], FluxArray[NoQualityIndex], "ko",MarkerSize=2, label="Unqualified")
    pl.legend(loc='best')
    pl.tight_layout()
    pl.savefig(outputfolder+"/GoodDatavsBadData.png")
    pl.close('all')


def GetLightCurve(filepath='',outputpath='',plot=False):
    '''
    Centroid are calculated by center of mass function from scipy
    Background are fitting by spline.
    '''
    print ("Running Predetermined Aperture")
    #extracting the starname
    starname = str(re.search('[0-9]{9}',filepath).group(0))
    Campaign = re.search('c[0-9]{2}',filepath).group(0)
    Campaign = int(Campaign[1:])


    #if short cadence data
    if "spd" in filepath:
      starname = starname+"_spd"


    #constructing the output folder path
    outputfolder = os.path.join(outputpath,starname)

    #read the FITS file
    try:
        FitsFile = fits.open(filepath,memmap=True) #opening the fits file
    except:
        raise Exception('Error opening the file')

    #make the directory if the directory does not exist
    TestPaths = [outputpath,outputfolder]
    for path in TestPaths:
        if not os.path.exists(path):
            os.system("mkdir %s" %(path))

    #extract the vital information from the fits file
    KeplerID = FitsFile[0].header['KEPLERID']
    print ("KEPLERID:", KeplerID)
    TotalDate = FitsFile[1].data['Time']
    TotalFlux = FitsFile[1].data['Flux']
    Quality = FitsFile[1].data['Quality']
    RA = FitsFile[0].header['RA_OBJ']
    Dec = FitsFile[0].header['DEC_OBJ']
    KepMag = FitsFile[0].header['Kepmag']
    print ("Kepler Magnitude:", KepMag)
    X = FitsFile[2].header['CRPIX1']  - 1.0 #-1 to account for the fact indexing begins at 0 in python
    Y = FitsFile[2].header['CRPIX2'] - 1.0

    #Download the picture from the region
    #DownloadImage(RA, Dec, outputfolder)



    #initiating array to collect values
    FluxArray = []
    X_Pos_Array = []
    Y_Pos_Array = []
    BkgArray = []
    FluxIndex = []


    if starname.endswith('_spd'):
        StarAperName = starname[:-4]
    else:
        StarAperName = starname

    AvgFlux = np.nanmean(TotalFlux, axis=0)


    print("Now finding the aperture")
    #make the aperture
    StdAper = FindAperture.Case1(AvgFlux,X,Y, StdCutOff=3.0)

    ApertureOutline(StdAper,AvgFlux, outputfolder, X,Y)

    #ApertureOutline(StdAper, AvgFlux, outputfolder, X, Y)



    if Campaign>8 and Campaign<12:
        for i in range(len(TotalFlux)):
            CurrentFrame = TotalFlux[i]
            BkgMedian = np.nanmedian(CurrentFrame)
            CurrentFrame[np.isnan(CurrentFrame)] = 0.0 #converting all nan to zero
            Flux = CurrentFrame*StdAper

            #Getting the total value of the Flux
            Background = np.sum(StdAper)*BkgMedian
            FluxValue = np.sum(Flux)
            FluxArray.append(FluxValue)
            BkgArray.append(Background)
            #0 means it is good
            #128 is for cosmic ray detection
            #16384 is for detector anamoly flag raised #sometime is raised without good reason

            QualityPass = bool(Quality[i]==0 or Quality[i]==16384 or Quality[i]==128 or np.isnan(BkgMedian))
            #QualityPass = bool(Quality[i]==0 or np.isnan(BkgMedian)) #Stricter quality pass

            if FluxValue>0 and QualityPass:
                YPos, XPos = measurements.center_of_mass(Flux)
                X_Pos_Array.append(XPos)
                Y_Pos_Array.append(YPos)
                FluxIndex.append(True)
            else:
                FluxIndex.append(False)
    else:
        #for campaign 0-8 and 12
        for i in range(len(TotalFlux)):
            CurrentFrame = TotalFlux[i]
            BkgMedian = np.nanmedian(CurrentFrame)
            CurrentFrame[np.isnan(CurrentFrame)] = 0.0 #converting all nan to zero
            CurrentDate = TotalDate[i]
            Flux = CurrentFrame*StdAper

            Background = np.sum(StdAper)*BkgMedian
            FluxValue = np.sum(Flux)
            FluxArray.append(FluxValue)
            BkgArray.append(Background)


            QualityPass = bool(Quality[i]==0)# or Quality[i]==16384 or Quality[i]==128 or np.isnan(BkgMedian))
            #QualityPass = bool(Quality[i]==0 or np.isnan(BkgMedian)) #Stricter quality pass


            if FluxValue>0 and QualityPass:
                YPos, XPos = measurements.center_of_mass(Flux)
                X_Pos_Array.append(XPos)
                Y_Pos_Array.append(YPos)
                FluxIndex.append(True)
            else:
                FluxIndex.append(False)



    RawFluxDiagram(Quality,TotalDate,FluxArray,outputfolder)
    FluxIndex = np.array(FluxIndex)
    FluxArray = np.array(FluxArray)
    BkgArray = np.array(BkgArray)

    FluxArray = FluxArray[FluxIndex]
    BkgArray = BkgArray[FluxIndex]
    DateArray = TotalDate[FluxIndex]
    X_Pos_Array =  np.array(X_Pos_Array)
    Y_Pos_Array = np.array(Y_Pos_Array)

    '''
    def moving_average(series, sigma=3):
        b = gaussian(39, sigma)
        average = filters.convolve1d(series, b/b.sum())
        var = filters.convolve1d(np.power(series-average,2), b/b.sum())
        return average, var

    _, var = moving_average(BkgArray)
    if "spd" in starname:
        factor = 0.75
    else:
        factor = 0.9
    spl =  UnivariateSpline(DateArray, BkgArray, w=factor/np.sqrt(var))
    SplEstimatedBkg = spl(DateArray)
    '''

    TempDate = np.copy(DateArray)
    TempBkg = np.copy(BkgArray)
    ChunkSize = 300
    for i in range(3):
          N = int(len(TempDate)/ChunkSize)
          Location = [int((i+0.5)*ChunkSize) for i in range(N)]
          knots = [TempDate[i] for i in Location]
          spl = spline(TempDate, TempBkg, knots, k=2)
          BkgEst = spl(TempDate)
          Residual = np.abs(TempBkg - BkgEst)
          Indices = Residual<3.0*np.std(Residual)
          TempDate = TempDate[Indices]
          TempBkg = TempBkg[Indices]


    SplEstimatedBkg = spl(DateArray)


    #saving diagnostic plot
    pl.figure(figsize=(20,10))
    pl.subplot(2,1,1)
    pl.plot(DateArray, BkgArray,"k.", MarkerSize=2)
    pl.plot(DateArray, SplEstimatedBkg,"g-",lw=2)
    pl.xlabel('Time (days)')
    pl.ylabel('Flux Count')
    pl.title('Background')

    pl.subplot(2,1,2)
    pl.plot(DateArray, FluxArray,"ko",MarkerSize=2)
    pl.title('Flux Reading')
    pl.xlabel('Time (days)')
    pl.ylabel('Flux Count')

    pl.suptitle(str(KeplerID)+" Diagnostic")
    pl.savefig(outputfolder+"/Background.png")
    pl.close('all')
    return DateArray, FluxArray, X_Pos_Array, Y_Pos_Array
