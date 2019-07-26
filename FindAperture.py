import numpy as np
import matplotlib.pyplot as pl
import matplotlib.colors as colors
import re
from astropy.io import fits
import os
from scipy.ndimage import convolve, label, measurements

from astropy.stats import gaussian_sigma_to_fwhm
from photutils.detection import IRAFStarFinder

from skimage.morphology import watershed
import itertools
import operator


def ApertureOutline(StdAper,KepMag, AvgFlux, outputfolder, starname, XPos, YPos):
    #find the outline and save the aperture in the relevant folder
    N = int(np.max(StdAper))
    l = []
    Xs = []
    Ys = []
    for i in range(1,N+1):
        TempAper = StdAper==i
        YCen, XCen = measurements.center_of_mass(TempAper*AvgFlux)
        Xs.append(XCen)
        Ys.append(YCen)
        ver_seg = np.where(TempAper[:,1:] != TempAper[:,:-1])
        hor_seg = np.where(TempAper[1:,:] != TempAper[:-1,:])


        for p in zip(*hor_seg):
            l.append((p[1], p[0]+1))
            l.append((p[1]+1, p[0]+1))
            l.append((np.nan,np.nan))

            for p in zip(*ver_seg):
                l.append((p[1]+1, p[0]))
                l.append((p[1]+1, p[0]+1))
                l.append((np.nan, np.nan))

    segments = np.array(l)

    #YCen, XCen = measurements.center_of_mass(StdAper*AvgFlux)
    pl.figure(figsize=(10,10))
    pl.imshow(AvgFlux,cmap='gray',norm=colors.PowerNorm(gamma=1./2.),interpolation='none')
    pl.colorbar()
    pl.plot(segments[:,0]-0.5, segments[:,1]-0.5, color=(1,0,0,.5), linewidth=3)
    pl.plot(XPos, YPos, "r+", markersize=4)
    for  i in range(len(Xs)):
        pl.plot(Xs[i],Ys[i],"g+", markersize=4)
    pl.title(starname+":"+str(KepMag))
    pl.gca().invert_yaxis()
    pl.axis('equal')
    pl.tight_layout()
    pl.savefig(outputfolder+"/"+starname+".png", bbox_inches='tight')
    pl.close('all')

def CompApertureOutline(StdAper1, StdAper2, KepMag, AvgFlux1, AvgFlux2, outputfolder, starname, XPos, YPos):
    #find the outline and save the aperture in the relevant folder
    YCen1, XCen1 = measurements.center_of_mass(StdAper1*AvgFlux1)
    ver_seg1 = np.where(StdAper1[:,1:] != StdAper1[:,:-1])
    hor_seg1 = np.where(StdAper1[1:,:] != StdAper1[:-1,:])

    l1 = []
    for p in zip(*hor_seg1):
        l1.append((p[1], p[0]+1))
        l1.append((p[1]+1, p[0]+1))
        l1.append((np.nan,np.nan))

    for p in zip(*ver_seg1):
        l1.append((p[1]+1, p[0]))
        l1.append((p[1]+1, p[0]+1))
        l1.append((np.nan, np.nan))

    segments1 = np.array(l1)

    YCen2, XCen2 = measurements.center_of_mass(StdAper2*AvgFlux2)
    ver_seg2 = np.where(StdAper2[:,1:] != StdAper2[:,:-1])
    hor_seg2 = np.where(StdAper2[1:,:] != StdAper2[:-1,:])

    l2 = []
    for p in zip(*hor_seg2):
        l2.append((p[1], p[0]+1))
        l2.append((p[1]+1, p[0]+1))
        l2.append((np.nan,np.nan))

    for p in zip(*ver_seg2):
        l2.append((p[1]+1, p[0]))
        l2.append((p[1]+1, p[0]+1))
        l2.append((np.nan, np.nan))

    segments2 = np.array(l2)

    #YCen, XCen = measurements.center_of_mass(StdAper*AvgFlux)
    pl.figure(figsize=(16,7))
    pl.subplot(121)
    pl.imshow(AvgFlux1,cmap='gray',norm=colors.PowerNorm(gamma=1./2.),interpolation='none')
    pl.colorbar()
    pl.plot(segments1[:,0]-0.5, segments1[:,1]-0.5, color=(1,0,0,.5), linewidth=3)
    pl.plot(XPos, YPos, "r+", markersize=10)
    pl.plot(XCen1,YCen1,"g+", markersize=10)
    pl.gca().invert_yaxis()
    pl.axis('equal')

    pl.subplot(122)
    pl.imshow(AvgFlux2,cmap='gray',norm=colors.PowerNorm(gamma=1./2.),interpolation='none')
    pl.colorbar()
    pl.plot(segments2[:,0]-0.5, segments2[:,1]-0.5, color=(1,0,0,.5), linewidth=3)
    pl.plot(XPos, YPos, "r+", markersize=4)
    pl.plot(XCen2,YCen2,"g+", markersize=4)
    pl.gca().invert_yaxis()
    pl.axis('equal')

    pl.suptitle(starname+":"+str(KepMag))
    pl.tight_layout(rect=[0, 0.03, 1, 0.95])
    pl.savefig(outputfolder+"/"+starname+"_comp.png", bbox_inches='tight')
    pl.close('all')


def CrowdedFieldSingle(AvgFlux,XPos,YPos, StdCutOff=3.0):
    '''returns the best aperture from the crowded field'''
    MedianFlux = np.median(AvgFlux)
    Std = np.std(np.nonzero(AvgFlux*(AvgFlux<MedianFlux)))
    Mask = AvgFlux>(MedianFlux+ StdCutOff*Std)
    MaskValue = MedianFlux + StdCutOff*Std

    MaskedImage = AvgFlux*Mask

    sigma_psf = 2.0
    #daofind = DAOStarFinder(fwhm=4., threshold=1.5*std)
    iraffind = IRAFStarFinder(threshold=MaskValue, fwhm=4.0, minsep_fwhm=0.01, sharplo=-20.0, sharphi=20.0,roundhi=20.0, roundlo=-20.0)
                             #fwhm=sigma_psf*gaussian_sigma_to_fwhm)

    sources = iraffind(MaskedImage)
    StarUpdLocation = np.zeros((len(AvgFlux),len(AvgFlux[0])))
    for i in range(len(sources)):
        StarUpdLocation[int(round(sources['ycentroid'][i],0)), int(round(sources['xcentroid'][i],0))] = i+1
    Apertures = watershed(-MaskedImage, StarUpdLocation, mask=Mask)

    Distance= 6.0
    for i in range(1,np.max(Apertures)):
        TempAper = Apertures==i
        TempImage = TempAper*MaskedImage
        YCen, XCen = measurements.center_of_mass(TempImage)
        TempDistance = ((XPos-XCen)**2+(YPos-YCen)**2)**0.5

        if TempDistance<Distance:
            Distance=TempDistance
            StdAper = TempAper

    if Distance>5.5:
        print ("Failed to find a good aperture")
        StdAper = Apertures==np.max(Apertures)
    return StdAper


def CrowdedFieldMultiple(AvgFlux,XPos,YPos, StdCutOff=3.0):
    '''returns the best aperture from the crowded field'''
    '''returns the best aperture from the crowded field'''
    MedianFlux = np.median(AvgFlux)
    Std = np.std(np.nonzero(AvgFlux*(AvgFlux<MedianFlux)))
    Mask = AvgFlux>(MedianFlux+ StdCutOff*Std)
    MaskValue = MedianFlux + StdCutOff*Std

    MaskedImage = AvgFlux*Mask


    sigma_psf = 2.0
    #daofind = DAOStarFinder(fwhm=4., threshold=1.5*std)
    iraffind = IRAFStarFinder(threshold=MaskValue, fwhm=4.0, minsep_fwhm=0.01, sharplo=-20.0, sharphi=20.0,roundhi=20.0, roundlo=-20.0)
                             #fwhm=sigma_psf*gaussian_sigma_to_fwhm)

    sources = iraffind(MaskedImage)
    StarUpdLocation = np.zeros((len(AvgFlux),len(AvgFlux[0])))
    for i in range(len(sources)):
        StarUpdLocation[int(round(sources['ycentroid'][i],0)), int(round(sources['xcentroid'][i],0))] = i+1

    Distance = np.exp(-1.05*(distance_transform_edt(Mask))**2)
    Apertures = watershed(-MaskedImage, StarUpdLocation, mask=Mask)
    return Apertures

def Case1(AvgFlux,X,Y, StdCutOff=3.0):
    #returns the maximum aperture

    NanIndex = np.where(np.isnan(AvgFlux))
    AvgFlux[NanIndex] = 0.0


    ExpectedFluxUnder = 1.1*np.nanmedian(AvgFlux)
    #find a standard Aperture

    AllAper = (AvgFlux>ExpectedFluxUnder)


    BkgAper = 1- AllAper
    BkgArray = AvgFlux[np.nonzero(BkgAper*AvgFlux)]
    BkgMedian = np.abs(np.nanmedian(BkgArray))

    BkgStd = np.nanstd(BkgArray)
    ExpectedFluxUnder = ExpectedFluxUnder+BkgStd*StdCutOff

    #return the biggest aperture
    TotalAper = 1.0*(AvgFlux>ExpectedFluxUnder)
    lw, num = measurements.label(TotalAper) # this numbers the different apertures distinctly
    area = measurements.sum(TotalAper, lw, index=np.arange(lw.max() + 1)) # this measures the size of the apertures
    TotalAper = area[lw].astype(int) # this replaces the 1s by the size of the aperture
    StdAper = (TotalAper >= np.max(TotalAper))*1  #backend process
    Distance = 5.5
    for i in range(1,np.max(TotalAper)+1):
        TempAper = (TotalAper==i)*1
        if np.sum(TempAper)>3:
            YCen, XCen = measurements.center_of_mass(TempAper*AvgFlux)
            TempDist = np.sqrt((X-XCen)**2+(Y-YCen)**2)
            if TempDist<Distance:
                Distance = TempDist
                StdAper = TempAper
    YCen, XCen = measurements.center_of_mass(StdAper*AvgFlux)
    if Distance>5.0:
        print ("Failed to find a good aperture")
    return StdAper



def Case2(AvgFlux,X,Y,Factor=2.2):
    #Use laplacian stencil to find all the stars in the scenes
    Median = np.abs(np.nanmedian(AvgFlux))
    ExpectedFluxUnder = Factor*Median
    #find a standard Aperture
    AllAper = (AvgFlux>ExpectedFluxUnder)
    AllAper, num = measurements.label(AllAper) # this numbers the different apertures distinctly

    Distance  = 6.0 #Unacceptable distance
    for i in range(1,num+1):
        TempAper = (AllAper == i)
        YCen, XCen = measurements.center_of_mass(TempAper*AvgFlux)
        TempDist = np.sqrt((X-XCen)**2+(Y-YCen)**2)
        if TempDist<Distance:
            Distance = TempDist
            StdAper = TempAper
    if Distance>5.0:
      raise Exception('Failed to find the aperture')
    return StdAper


def Case3(AvgFlux,X,Y,StdCutOff=3.0):
    #Use laplacian stencil to find all the stars in the scenes
    Median = np.nanmedian(AvgFlux)

    #find a background
    BkgAper = (AvgFlux<Median)

    FluxArray = AvgFlux[np.nonzero(BkgAper*AvgFlux)]
    Std = np.std(FluxArray)

    AllAper = AvgFlux>(Std*StdCutOff+Median)
    AllAper, num = measurements.label(AllAper) # this numbers the different apertures distinctly

    Distance  = 10.0 #Unacceptable distance
    for i in range(1,num+1):
        TempAper = (AllAper == i)
        YCen, XCen = measurements.center_of_mass(TempAper*AvgFlux)
        TempDist = np.sqrt((X-XCen)**2+(Y-YCen)**2)
        if TempDist<Distance:
            Distance = TempDist
            StdAper = TempAper
    if Distance>5:
        print ("Failed to find a good aperture")
    return StdAper



def Case5(AvgFlux, X,Y,Spacing=1):
    Spacing = int(Spacing)
    Xint, Yint = [int(round(X,0)), int(round(Y,0))]
    BkgVal = np.median(AvgFlux)
    XValues = np.arange(Xint-Spacing,Xint+Spacing+1,1)
    YValues = np.arange(Yint-Spacing,Yint+Spacing+1,1)

    ReferenceValue = 0

    Dist_tolerance = 0.75
    for i,j in list(itertools.product(XValues,YValues)):
        try:
            TempAper2_2 = np.zeros((len(AvgFlux),len(AvgFlux[0])))
            TempAper2_2[i:i+2, j:j+2] = 1
            Num = np.sum(TempAper2_2)
            Signal = np.sum(AvgFlux*TempAper2_2)-Num*BkgVal
            Y_Cen, X_Cen = measurements.center_of_mass(AvgFlux*TempAper2_2)
            Distance = np.sqrt((X- X_Cen)**2+(Y- Y_Cen)**2)
            if Distance<Dist_tolerance:
                Value2_2 = Signal/np.sqrt(Signal+ (Num+1)*BkgVal)
            else:
                Value2_2 = 1
        except:
            Value2_2 = 1

        try:
            TempAper2_3 = np.zeros(len(AvgFlux[0])*len(AvgFlux)).reshape(len(AvgFlux),len(AvgFlux[0]))
            TempAper2_3[i:i+2, j:j+3] = 1
            Num = np.sum(TempAper2_3)
            Signal = np.sum(AvgFlux*TempAper2_3)-Num*BkgVal
            Y_Cen, X_Cen = measurements.center_of_mass(AvgFlux*TempAper2_3)
            Distance = np.sqrt((X- X_Cen)**2+(Y- Y_Cen)**2)
            if Distance<Dist_tolerance:
                Value2_3 = Signal/np.sqrt(Signal+(Num+1)*BkgVal)
            else:
                Value2_3 = 2
        except:
            Value2_3 = 2

        try:
            TempAper3_2 = np.zeros(len(AvgFlux[0])*len(AvgFlux)).reshape(len(AvgFlux),len(AvgFlux[0]))
            TempAper3_2[i:i+3, j:j+2] = 1
            Num = np.sum(TempAper3_2)
            Value3_2 = np.sum(AvgFlux*TempAper3_2)-Num*BkgVal
            Y_Cen, X_Cen = measurements.center_of_mass(AvgFlux*TempAper3_2)
            Distance = np.sqrt((X- X_Cen)**2+(Y- Y_Cen)**2)
            if Distance<Dist_tolerance:
                Value3_2 = Signal/np.sqrt(Signal+(Num+1)*BkgVal)
            else:
                Value3_2 = 3
        except:
            Value3_2 = 3

        try:
            TempAper3_3 = np.zeros(len(AvgFlux[0])*len(AvgFlux)).reshape(len(AvgFlux),len(AvgFlux[0]))
            TempAper3_3[i:i+3, j:j+3] = 1
            Num = np.sum(TempAper3_3)
            Signal = np.sum(AvgFlux*TempAper3_3)-Num*BkgVal
            Y_Cen, X_Cen = measurements.center_of_mass(AvgFlux*TempAper3_3)
            Distance = np.sqrt((X- X_Cen)**2+(Y- Y_Cen)**2)
            if Distance<Dist_tolerance:
                Value3_3 = Signal/np.sqrt(Signal+(Num+1)*BkgVal)
            else:
                Value3_3 = 4
        except:
            Value3_3 = 4

        #star like shaped with five selection
        try:
            TempAper_Star = np.zeros(len(AvgFlux[0])*len(AvgFlux)).reshape(len(AvgFlux),len(AvgFlux[0]))
            TempAper_Star[i+1:i+2, j:j+3] = 1
            TempAper_Star[i:i+3, j+1:j+2] = 1
            Num = np.sum(TempAper_Star)
            Signal = np.sum(AvgFlux*TempAper_Star)-Num*BkgVal
            Y_Cen, X_Cen = measurements.center_of_mass(AvgFlux*TempAper_Star)
            Distance = np.sqrt((X- X_Cen)**2+(Y- Y_Cen)**2)
            if Distance<Dist_tolerance:
                Value_Star = Signal/np.sqrt(Signal+(Num+1)*BkgVal)
            else:
                Value_Star = 5
        except:
            Value_Star = 5

        #See which one is the best fit
        Values = np.array([Value2_2, Value2_3, Value3_2, Value3_3, Value_Star])
        MaxValue = max(Values)
        if MaxValue>ReferenceValue:
            ReferenceValue = MaxValue
            RefX, RefY = [i,j]
            TypeAperture = np.where(MaxValue == Values)[0][0]

    if ReferenceValue<7:
        #Find suitable 4 by 4 aperture based on distance
        print ("-"*50)
        print ("Failed to find a good aperture")
        print ("-"*50)

        #Aperture just based on the distance
        RefX, RefY = [Xint, Yint]
        DistanceRef = 5
        for i,j in list(itertools.product(XValues,YValues)):
          try:
            TempAper2_2 = np.zeros(len(AvgFlux[0])*len(AvgFlux)).reshape(len(AvgFlux),len(AvgFlux[0]))
            TempAper2_2[i:i+2, j:j+2] = 1
            Y_Cen, X_Cen = measurements.center_of_mass(AvgFlux*TempAper2_2)
            Distance = np.sqrt((X- X_Cen)**2+(Y- Y_Cen)**2)
            if Distance<DistanceRef:
                RefX,RefY = [i,j]
                DistanceRef = Distance
          except:
            pass
        TypeAperture = 0


    Aperture = np.zeros(len(AvgFlux[0])*len(AvgFlux)).reshape(len(AvgFlux),len(AvgFlux[0]))
    i,j = [RefX, RefY]
    if TypeAperture == 0:
        Aperture[i:i+2,j:j+2] = 1
    elif TypeAperture == 1:
        Aperture[i:i+2, j:j+3] = 1
    elif TypeAperture == 2:
        Aperture[i:i+3, j:j+2] = 1
    elif TypeAperture == 3:
        Aperture[i:i+3, j:j+3] = 1
    elif TypeAperture == 4:
        Aperture[i+1:i+2, j:j+3] = 1
        Aperture[i:i+3, j+1:j+2] = 1
    else:
        raise Exception('Error finding a good aperture')
    return Aperture



def Case6(AvgFlux, X,Y,Spacing=4):
    '''Same thing but for the brighter star'''

    '''grid search spacing around the known position of the star'''

    Spacing = int(Spacing)
    Xint, Yint = [int(round(X,0)), int(round(Y,0))]
    BkgVal = 35.0#np.median(AvgFlux)

    XValues = np.arange(Xint-Spacing,Xint+Spacing+1,1)
    YValues = np.arange(Yint-Spacing,Yint+Spacing+1,1)

    Dist_tolerance = 1.0 #Brighter Stars should be precisely located
    RefValue = 1.1
    ReadNoise = 15
    for i,j in list(itertools.product(XValues,YValues)):
        for m,n in list(itertools.product([3,4,5,6],[3,4,5,6])):
            try:
                TempAper = np.zeros(len(AvgFlux[0])*len(AvgFlux)).reshape(len(AvgFlux),len(AvgFlux[0]))
                TempAper[i:i+m, j:j+n] = 1
                Num = np.sum(TempAper)
                Signal = np.sum(AvgFlux*TempAper)-Num*BkgVal
                Y_Cen, X_Cen = measurements.center_of_mass(AvgFlux*TempAper)
                Distance = np.sqrt((X- X_Cen)**2+(Y- Y_Cen)**2)
                if Distance<Dist_tolerance:
                    Value = Signal/np.sqrt(Signal+(Num)*BkgVal+ReadNoise*Num)
                    if Value>RefValue:
                        RefValue = Value
                        Aperture = TempAper
            except:
                pass

    if RefValue == 1.1:
        print ("Failed to find a good aperture")
        Aperture = TempAper

    return Aperture

def VanderburgAper(AvgFlux,X,Y,BkgValue=0):

    #print "The median value is:", np.nanmedian(AvgFlux)
    BkgMedian = np.abs(np.nanmedian(AvgFlux))
    AvgFlux[np.isnan(AvgFlux)]=BkgMedian

    if BkgValue != 0:
        Background = BkgValue
    else:
        Background = np.abs(BkgMedian)
    ReadNoise = 25.0
    rad = np.linspace(1,8,500)
    XInt = np.arange(len(AvgFlux[0]))
    YInt = np.arange(len(AvgFlux))
    XX,YY = np.meshgrid(XInt, YInt)

    DistanceTol = 0.75
    ReferenceValue = 0
    for i in range(len(rad)):
        try:
            Mask = np.sqrt((XX-X)**2+(YY-Y)**2)<rad[i]
            YCen, XCen = measurements.center_of_mass(Mask*AvgFlux)
            Distance = np.sqrt((X-XCen)**2+(Y-YCen)**2)

            N = np.sum(Mask)
            Signal = np.sum(Mask*AvgFlux)
            SNR  = (Signal - N*Background)/np.sqrt(Signal+ N*Background+ N*(ReadNoise))
            if SNR>ReferenceValue and Distance<DistanceTol:
                ReferenceValue = SNR
                StdAper = np.copy(Mask)
        except:
            '''If the radius exceeds'''
            pass
    if ReferenceValue==0:
        print ("Failed to find a good aperture")
        StdAper = np.ones(len(AvgFlux)*len(AvgFlux[0])).reshape(len(AvgFlux),len(AvgFlux[0]))
    return StdAper

def FindAperture(filepath='',outputpath='',SubFolder='',CampaignNum='1'):
    '''
    Centroid are calculated by center of mass function from scipy
    Two different apertures
    '''
    outputfolder = outputpath+'/'+SubFolder

    #extracting the starname
    starname = str(re.search('[0-9]{9}',filepath).group(0))

    #read the FITS file
    try:
        FitsFile = fits.open(filepath,memmap=True) #opening the fits file
    except:
        raise Exception('Error opening the file')


    #extract the vital information from the fits file
    TotalDate = np.array(FitsFile[1].data['Time'])
    TotalFlux = FitsFile[1].data['Flux']
    Quality = FitsFile[1].data['Quality']
    KepMag = FitsFile[0].header['Kepmag']
    X = FitsFile[2].header['CRPIX1']  - 1.0 #-1 to account for the fact indexing begins at 0 in python
    Y = FitsFile[2].header['CRPIX2'] - 1.0
    FitsFile.close()

    Index = np.where(Quality==0)
    GoodFlux = np.array(operator.itemgetter(*Index[0])(TotalFlux))
    TotalDate = np.array(operator.itemgetter(*Index[0])(TotalDate))

    if CampaignNum>8 and CampaignNum<12:
        AvgFlux = np.nanmedian(GoodFlux, axis=0)
        MedianValue = np.nanmedian(AvgFlux)-0.5
        AvgFlux[np.isnan(AvgFlux)] = MedianValue

        if "1_lpd" in filepath:
            starnameTxt = starname+"_1"
        else:

            starnameTxt = starname+"_2"
        if KepMag<16:
            #StdAper = Case1(AvgFlux,X,Y,StdCutOff=2.0)*1 #use standard deviation of Background to cut it off
            StdAper = Case2(AvgFlux,X,Y,Factor=1.5) #Use the median value in the aperture
            #StdAper = Case3(AvgFlux,X,Y,StdCutOff=0.75) #Use the flux value of the star as cut off
            #StdAper = Case5(AvgFlux, X, Y, Spacing=3)*1 #
        else:
            #StdAper = Case5(AvgFlux, X, Y, Spacing=3)*1 #
            StdAper = Case2(AvgFlux,X,Y,Factor=1.5)
        ApertureOutline(StdAper,KepMag, AvgFlux, outputfolder, starnameTxt, X, Y)
        np.savetxt(outputfolder+"/"+starnameTxt+".txt",StdAper)

    else:
         AvgFlux = np.nanmedian(GoodFlux, axis=0)
         AvgFlux[np.isnan(AvgFlux)] = np.nanmedian(AvgFlux) #convert nan to the medians
         if KepMag<17:
             print ("Trying the case here")
             #StdAper = CrowdedFieldSingle(AvgFlux,X,Y, StdCutOff=7.5)
             #StdAper = Case1(AvgFlux,X,Y, StdCutOff=15.0)
             #StdAper = VanderburgAper(AvgFlux,X,Y)#, BkgValue=50.0) #If no BkgValue is provided the median value is used
             StdAper = Case2(AvgFlux,X,Y,Factor=4.0) #Use Van Eylen method for finding aperture

             #Helpful for bright stars convolving case
             #StdAper = OldCase4(AvgFlux1,X,Y)*1 #use standard deviation of Background to cut it off

             #StdAper = Case2(AvgFlux,X,Y,MedianTimes=1.25) #Use the median value in the aperture
             #StdAper = Case3(AvgFlux,X,Y,StdCutOff=3.0) #Use the flux value of the star as cut off
             #StdAper = Case4(AvgFlux, X, Y, Spacing=2)*1 #

             #print "Stage 2"
         else:
             #StdAper = Case5(AvgFlux, X, Y, Spacing=4)*1 #
             StdAper = VanderburgAper(AvgFlux,X,Y, BkgValue=5.0)

         ApertureOutline(StdAper,KepMag, AvgFlux, outputfolder, starname, X, Y)
         np.savetxt(outputfolder+"/"+starname+".txt",StdAper)


    SummaryFile = open(outputfolder+".csv",'a')
    SummaryFile.write(starname+",1,0 \n")
    SummaryFile.close()














########################################################################################################################################
####Backup code########################################################################################################################

def FindApertureBackUp(filepath='',outputpath='',SubFolder='',CampaignNum='1'):
    '''
    Centroid are calculated by center of mass function from scipy
    Two different apertures
    '''
    outputfolder = outputpath+'/'+SubFolder

    #extracting the starname
    starname = str(re.search('[0-9]{9}',filepath).group(0))

    #read the FITS file
    try:
        FitsFile = fits.open(filepath,memmap=True) #opening the fits file
    except:
        raise Exception('Error opening the file')


    #extract the vital information from the fits file
    TotalDate = np.array(FitsFile[1].data['Time'])
    TotalFlux = FitsFile[1].data['Flux']
    Quality = FitsFile[1].data['Quality']
    KepMag = FitsFile[0].header['Kepmag']
    X = FitsFile[2].header['CRPIX1']  - 1.0 #-1 to account for the fact indexing begins at 0 in python
    Y = FitsFile[2].header['CRPIX2'] - 1.0
    FitsFile.close()

    Index = np.where(Quality==0)
    GoodFlux = np.array(operator.itemgetter(*Index[0])(TotalFlux))
    TotalDate = np.array(operator.itemgetter(*Index[0])(TotalDate))

    if CampaignNum>8:
        AvgFlux = np.nanmedian(GoodFlux, axis=0)
        MedianValue = np.nanmedian(AvgFlux)-0.5
        AvgFlux[np.isnan(AvgFlux)] = MedianValue

        if "1_lpd" in filepath:
            starnameTxt = starname+"_1"
        else:

            starnameTxt = starname+"_2"
        if KepMag<16:
            StdAper = Case1(AvgFlux,X,Y,StdCutOff=5.0)*1 #use standard deviation of Background to cut it off
            #StdAper = Case2(AvgFlux,X,Y,MedianTimes=2.0) #Use the median value in the aperture
            #StdAper = Case3(AvgFlux,X,Y,StdCutOff=0.75) #Use the flux value of the star as cut off
            #StdAper = Case5(AvgFlux, X, Y, Spacing=3)*1 #
        else:
            StdAper = Case5(AvgFlux, X, Y, Spacing=3)*1 #
        ApertureOutline(StdAper,KepMag, AvgFlux, outputfolder, starnameTxt, X, Y)
        np.savetxt(outputfolder+"/"+starnameTxt+".txt",StdAper)

    else:
         #Two aperture method
         print ("Here")
         DateHalf = (max(TotalDate)+min(TotalDate))/2.0

         FirstHalf = TotalDate<DateHalf
         SecondHalf = TotalDate>DateHalf

         AvgFlux1 = np.nanmedian(GoodFlux[FirstHalf], axis=0)
         MedianValue1 = np.nanmedian(GoodFlux[FirstHalf])-0.5
         AvgFlux1[np.isnan(AvgFlux1)] = MedianValue1

         AvgFlux2 = np.nanmedian(GoodFlux[SecondHalf], axis=0)
         MedianValue2 = np.nanmedian(GoodFlux[SecondHalf])-0.5
         AvgFlux2[np.isnan(AvgFlux2)] = MedianValue2


         if KepMag<17:
             #StdAper1 = CrowdedFieldSingle(AvgFlux1,X,Y, StdCutOff=2.0)
             #StdAper2 = CrowdedFieldSingle(AvgFlux2,X,Y, StdCutOff=2.0)


             #StdAper1 = Case1(AvgFlux1,X,Y, StdCutOff=50.0)
             #StdAper2 = Case1(AvgFlux2,X,Y, StdCutOff=30.0)

             StdAper1 = VanderburgAper(AvgFlux1,X,Y)
             StdAper2 = VanderburgAper(AvgFlux2,X,Y)


             #StdAper1 = Case2(AvgFlux1,X,Y,Factor=0.0005)*1 #use standard deviation of Background to cut it off
             #StdAper2 = Case2(AvgFlux2,X,Y,Factor=0.0005)*1

             #Helpful for bright stars convolving case
             #StdAper1 = OldCase4(AvgFlux1,X,Y)*1 #use standard deviation of Background to cut it off
             #StdAper2 = OldCase4(AvgFlux2,X,Y)*1

             #StdAper = Case2(AvgFlux,X,Y,MedianTimes=1.25) #Use the median value in the aperture
             #StdAper = Case3(AvgFlux,X,Y,StdCutOff=1.0) #Use the flux value of the star as cut off
             #StdAper = Case4(AvgFlux, X, Y, Spacing=2)*1 #

             #print "Stage 2"
         else:
             StdAper1 = Case5(AvgFlux1, X, Y, Spacing=4)*1 #
             StdAper2 = Case5(AvgFlux2, X, Y, Spacing=4)*1
         ApertureOutline(StdAper1,KepMag, AvgFlux1, outputfolder, starname+"_1", X, Y)
         ApertureOutline(StdAper2,KepMag, AvgFlux2, outputfolder, starname+"_2", X, Y)
         CompApertureOutline(StdAper1, StdAper2, KepMag, AvgFlux1, AvgFlux2, outputfolder, starname, X, Y)

         np.savetxt(outputfolder+"/"+starname+"_1.txt",StdAper1)
         np.savetxt(outputfolder+"/"+starname+"_2.txt",StdAper2)

    SummaryFile = open(outputfolder+".csv",'a')
    SummaryFile.write(starname+",1,0 \n")
    SummaryFile.close()

######################Old methods for finding aperture##############################
def OldCase1(AvgFlux):
    ExpectedFluxUnder = np.median(AvgFlux)

    #find a standard Aperture
    StdAper = (AvgFlux>ExpectedFluxUnder)
    lw, num = measurements.label(StdAper) # this numbers the different apertures distinctly
    area = measurements.sum(StdAper, lw, index=np.arange(lw.max() + 1)) # this measures the size of the apertures
    StdAper = area[lw].astype(int) # this replaces the 1s by the size of the aperture
    StdAper = (StdAper >= np.max(StdAper))*1 #make the standard aperture as 1.0

    #Finding the background aperture
    BkgAper = 1.0 - StdAper


    BkgFrame = (BkgAper*AvgFlux)
    BkgFrame = BkgFrame[np.nonzero(BkgFrame)]
    BkgStd = np.std(BkgFrame)
    BkgMedian = np.median(BkgFrame) #negative values for background are sometimes seen, which means that will be added to the flux values rather than subtracted
    Sigma = 5.0 #Usual value is 5
    CutoffLower = BkgMedian - Sigma*BkgStd #5 sigma cutoff for excluding really unusual pixel


    #New method
    BkgFrame = BkgFrame[np.nonzero((BkgFrame>CutoffLower)*1.0)]
    #BkgNewMean = np.median(BkgFrame)
    BkgNewMean = np.abs(np.median(BkgFrame))
    BkgNewStd = np.std(BkgFrame)

    Sigma = 2.0 ###Important for determining the aperture
    ExpectedFluxUnder = BkgNewMean+Sigma*BkgNewStd+15.0 #15.0 to consider the case where the background is really small


    #find a standard Aperture
    StdAper = 1.0*(AvgFlux>ExpectedFluxUnder)
    lw, num = measurements.label(StdAper) # this numbers the different apertures distinctly
    area = measurements.sum(StdAper, lw, index=np.arange(lw.max() + 1)) # this measures the size of the apertures
    StdAper = area[lw].astype(int) # this replaces the 1s by the size of the aperture
    StdAper = (StdAper >= np.max(StdAper))*1 #
    return StdAper

def OldCase2(AvgFlux):
    ExpectedFluxUnder = 2*np.median(AvgFlux)

    StdAper = (AvgFlux>ExpectedFluxUnder)
    lw, num = measurements.label(StdAper) # this numbers the different apertures distinctly
    area = measurements.sum(StdAper, lw, index=np.arange(lw.max() + 1)) # this measures the size of the apertures
    StdAper = area[lw].astype(int) # this replaces the 1s by the size of the aperture
    StdAper = (StdAper >= np.max(StdAper))*1 #make the standard aperture as 1.0

    return StdAper

def OldCase3(AvgFlux):
    ExpectedFluxUnder = 175
    StdAper = (AvgFlux>ExpectedFluxUnder)
    lw, num = measurements.label(StdAper) # this numbers the different apertures distinctly
    area = measurements.sum(StdAper, lw, index=np.arange(lw.max() + 1)) # this measures the size of the apertures
    StdAper = area[lw].astype(int) # this replaces the 1s by the size of the aperture
    StdAper = (StdAper >= np.max(StdAper))*1 #make the standard aperture as 1.0
    return StdAper

def OldCase4(AvgFlux, X, Y):
    #Convolve with a laplacian
    LaplacianStencil = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    Laplacian = convolve(AvgFlux, LaplacianStencil)
    StdAper = (Laplacian<-10)
    lw, num = measurements.label(StdAper) # this numbers the different apertures distinctly
    area = measurements.sum(StdAper, lw, index=np.arange(lw.max() + 1)) # this measures the size of the apertures
    StdAper = area[lw].astype(int) # this replaces the 1s by the size of the aperture
    StdAper = (StdAper >= np.max(StdAper))*1
    pl.figure(figsize=(16,7))
    pl.subplot(121)
    pl.imshow(AvgFlux,cmap='gray',norm=colors.PowerNorm(gamma=1./2.),interpolation='none')
    pl.plot(X,Y,"ko")
    pl.colorbar()
    pl.subplot(122)
    pl.imshow(StdAper)
    pl.colorbar()
    pl.plot(X,Y,"ko")
    pl.show()


    return StdAper
