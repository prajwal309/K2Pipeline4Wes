import os
import matplotlib.pyplot as pl
import numpy as np
from lmfit import minimize, Parameters
import glob

# pipeline files
from auxiliaries import *
from scipy.ndimage import filters
from scipy.stats.stats import pearsonr
from scipy.integrate import quad
from scipy.interpolate import LSQUnivariateSpline as spline
from scipy.interpolate import UnivariateSpline
from scipy import signal
from scipy.ndimage import filters
import os

def moving_average(series, sigma=3.0):
    b = signal.gaussian(24, sigma)
    average = filters.convolve1d(series, b/b.sum())
    var = filters.convolve1d(np.power(series-average,2), b/b.sum())
    return average, var


def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in range(0, len(l), n):
        yield list(l[i:i+n])

def median_filter(time,data,binsize=30):
  # do a running median filter dividing all data points by the median of their immediate surroundings
  i = 0
  data_filtered = []
  while i < len(time):
	  bin_begin = int(max(0, (i - binsize/2)))
	  bin_end = int(min(len(time),(i+binsize/2)))
	  the_bin = data[bin_begin:bin_end]
	  the_bin = sorted(the_bin)
	  median = np.median(the_bin) #[len(the_bin)/2]
	  data_filtered.append(data[i]/median)
	  i = i + 1
  return data_filtered


def NewMedianFilter(time,flux, binsize=24):

  # do a running median filter dividing all data points by the median of their immediate surroundings
  N = int(len(time)/binsize)
  MedianArray = []
  for i in range(N+1):
     median = np.median(flux[i*binsize:(i+1)*binsize])
     if i<N:
         for j in range(binsize):
             MedianArray.append(median)
     else:
         for j in range(len(time)-N*binsize):
             MedianArray.append(median)
  return MedianArray


def K2SFF_VJ14(time,flux, Xc, Yc,CmpStr,starname='',outputpath='',chunksize=75,  PolyDeg=2):

  #Location to store file
  outputfolder = os.path.join(outputpath,str(starname))

  #flux = np.array(median_filter(time, flux, 49.))
  CmpStr=CmpStr[1:]
  PointingFiles = glob.glob("PointingVectors/%s*Processed*.txt" %CmpStr)

  ChunkSize = 72#chunk every 1.5 day
  TempTime = np.copy(time)
  TempFlux = np.copy(flux)

  for k in range(3):
     N = int(len(TempTime)/ChunkSize)
     Location = [int((i+0.5)*ChunkSize) for i in range(N)]
     knots = [TempTime[i] for i in Location]
     spl = spline(TempTime, TempFlux, knots[1:-1], k=2)  #-1 to -1 to end overfitting at the end points
     DetrendedFlux = spl(TempTime)
     Residual = np.abs(TempFlux -DetrendedFlux)
     Indices = Residual<3.5*np.std(Residual)
     TempFlux = TempFlux[Indices]
     TempTime = TempTime[Indices]
     DetrendedFlux = DetrendedFlux[Indices]

  TimeDetrend = spl(time)
  TimeDetrendedFlux = flux/TimeDetrend

  pl.figure(figsize=(15,5))
  pl.clf()
  pl.plot(time, flux, "ko")
  pl.plot(time,TimeDetrend, "r-", lw=2)
  pl.savefig(outputfolder+"/InitialTimeTrend.png")
  pl.close('all')


  #detrend in time again
  ChunkSize = 100#chunk every 1.5 day
  OriginalTimeDetrend = np.copy(TimeDetrendedFlux)
  TempTime = np.copy(time)
  TempFlux = np.copy(TimeDetrendedFlux)
  for k in range(3):
     N = int(len(TempTime)/ChunkSize)
     Location = [int((i+0.5)*ChunkSize) for i in range(N)]
     knots = [TempTime[i] for i in Location]
     spl = spline(TempTime, TempFlux, knots[1:-1], k=2)  #-1 to -1 to end overfitting at the end points
     DetrendedFlux = spl(TempTime)
     Residual = np.abs(TempFlux -DetrendedFlux)
     Indices = Residual<3.5*np.std(Residual)
     TempFlux = TempFlux[Indices]
     TempTime = TempTime[Indices]

  TimeDetrend = spl(time)
  TimeDetrendedFlux = OriginalTimeDetrend/TimeDetrend

  pl.figure(figsize=(15,5))
  pl.clf()
  pl.plot(time, OriginalTimeDetrend, "ko")
  pl.plot(time,TimeDetrend, "r-", lw=2)
  pl.savefig(outputfolder+"/SecondTimeTrend.png")
  pl.close('all')


  #Calculate the correlation coefficient with its own centroid positioning

  Xc = Xc - np.median(Xc)
  Yc = Yc - np.median(Yc)

  Mask = np.where((abs(Xc - np.mean(Xc)) <= 3*np.std(Xc)) & (abs(Yc - np.mean(Yc)) <= 3*np.std(Yc)))

  time = time[Mask]
  flux = flux[Mask]
  Xc = Xc[Mask]
  Yc = Yc[Mask]
  TimeDetrendedFlux = TimeDetrendedFlux[Mask]

  ChunkSize = 300#small chunksize and large chunksize are two different things
  time_arc_chunks = list(chunks(time,ChunkSize))
  flux_arc_chunks = list(chunks(flux,ChunkSize))
  Xc_arc_chunk = list(chunks(Xc,ChunkSize))
  Yc_arc_chunks = list(chunks(Yc,ChunkSize))

  #if the last chunk is really small
  if len(time_arc_chunks[-1])<ChunkSize/2.0:
      time_arc_chunks[-2].extend(time_arc_chunks[-1])
      time_arc_chunks.pop()
      flux_arc_chunks[-2].extend(flux_arc_chunks[-1])
      flux_arc_chunks.pop()
      Xc_arc_chunk[-2].extend(Xc_arc_chunk[-1])
      Xc_arc_chunk.pop()
      Yc_arc_chunks[-2].extend(Yc_arc_chunks[-1])
      Yc_arc_chunks.pop()



  ArcLength = []
  counter = 0

  for t_300,X_300,Y_300,f_300  in zip(time_arc_chunks,Xc_arc_chunk,Yc_arc_chunks,flux_arc_chunks):
      Centroids =  np.vstack([X_300,Y_300])
      Covar = np.cov(Centroids)
      [e1, v1] = np.linalg.eig(Covar)

      Centroids_Rot = np.dot(v1.T,Centroids)
      Perp1, Perp2 = Centroids_Rot[0,:], Centroids_Rot[1,:]

      if np.std(Perp1)>np.std(Perp2):
          X_Transformed = Perp1
          Y_Transformed = Perp2
      else:
          X_Transformed = Perp2
          Y_Transformed = Perp1

      #FIT A FIFTH ORDER POLYNOMIAL
      TempXTransformed = np.copy(X_Transformed)
      TempYTransformed = np.copy(Y_Transformed)

      params = np.polyfit(TempXTransformed, TempYTransformed,5)
      XXX = np.linspace(min(TempXTransformed),max(TempXTransformed),1000)
      YYY = np.polyval(params, XXX)



      #Calculate Arc Length
      Mult = np.arange(len(params)-1,0,-1)
      NewParams = params[:-1]*Mult

      #This helps you evaluate the integrand
      def Integrand(x,params):
          return np.sqrt(1+(np.polyval(params,x))**2)


      TempArcLength = []
      #Calculate Arclength for every coordinates
      for XVal in X_Transformed:
          tempArc,_ = quad(Integrand,0,XVal,args=(NewParams),epsabs=1e-8)
          TempArcLength.append(tempArc)

      TempArcLength = np.array(TempArcLength).astype(np.float32)
      ArcLength.extend(TempArcLength)


  ArcLength = np.array(ArcLength)*3.98
  params = np.polyfit(ArcLength,TimeDetrendedFlux,5)
  FluxPred = np.polyval(params, ArcLength)

  XXX = np.linspace(min(ArcLength), max(ArcLength), 1000)
  YYY = np.polyval(params, XXX)

  pl.figure()
  pl.plot(ArcLength, TimeDetrendedFlux, "ko")
  pl.plot(XXX,YYY,"r-",lw=2)
  pl.xlabel("Arclength (Arcseconds)")
  pl.ylabel("Relative Flux")
  pl.savefig(outputfolder+"/RelativeFlux_Self_ArcLength.png")

  StartStd =  np.std(TimeDetrendedFlux-FluxPred)
  SelectedFile = "Self"
  for FileName in PointingFiles:
      ReadFile = np.loadtxt(FileName,delimiter=',',skiprows=1)
      t_pointing = ReadFile[:,0].astype(np.float32)
      Arc_pointing = ReadFile[:,2]*3.98 #convert into arc seconds

      #Need to interpolate because the read time does not match exactly
      TargetArc = np.interp(time, t_pointing, Arc_pointing)
      params = np.polyfit(TargetArc,TimeDetrendedFlux,5)
      FluxPred = np.polyval(params, TargetArc)
      StdCurrent = np.std(TimeDetrendedFlux-FluxPred)

      if StartStd>StdCurrent:
          SelectedFile = FileName
          StartStd = StdCurrent
          TargetArc = np.interp(time, t_pointing, Arc_pointing)
          XXX = np.linspace(min(TargetArc), max(TargetArc), 1000)
          YYY = np.polyval(params, XXX)
          pl.figure()
          pl.plot(TargetArc, TimeDetrendedFlux, "ko")
          pl.plot(XXX,YYY,"r-",lw=2)
          pl.xlabel("Arclength (Arcseconds)")
          pl.ylabel("Relative Flux")
          pl.savefig(outputfolder+"/RelativeFlux_ArcLength.png")
          pl.close('all')


  print ("The Selected file is ", SelectedFile)

  if SelectedFile=="Self":
      TargetArc = np.copy(ArcLength)

  ds = np.diff(TargetArc)
  dt = np.diff(time)
  dV = ds/dt + 50.0


  TempTime = np.copy(time[1:])
  TempdV = np.copy(dV)





  ChunkSize = 72 #chunk every day or so
  for k in range(3):
     N = int(len(TempTime)/ChunkSize)
     Location = [int((i+0.5)*ChunkSize) for i in range(N)]
     knots = [TempTime[i] for i in Location]
     spl_dv = spline(TempTime, TempdV, knots, k=2)
     Detrended_dV = spl_dv(TempTime)
     Residual = np.abs(TempdV -Detrended_dV)
     Indices = Residual<3.0*np.std(Residual)
     TempTime = TempTime[Indices]
     TempdV = TempdV[Indices]

  #There is a time trend in dV. Remove it
  Detrended_dV = spl_dv(time[1:])
  dV = np.abs(dV/Detrended_dV) -1.0
  Std = np.std(dV)
  ThrusterIndices = np.abs(dV)>2.0*Std
  ThrusterIndices = np.concatenate([np.array([False]),ThrusterIndices])
  Locations = np.where(ThrusterIndices)[0]

  fig, (ax1, ax2, ax3) = pl.subplots(3, figsize=(15, 7), dpi=80, facecolor='w', edgecolor='k', sharex=True)
  ax1.plot(time,Xc,"bo")
  ax2.plot(time,Yc,"go")
  ax3.plot(time,TimeDetrendedFlux,"ko")
  for Location in Locations:
      ax1.axvline(x=time[Location],c="red",ymin=-10.0,ymax=10.0, clip_on=True)
      ax2.axvline(x=time[Location],c="red",ymin=-10.0,ymax=10.0, clip_on = True)
      ax3.axvline(x=time[Location],c="red",ymin=-10.0,ymax=10.0, clip_on = True)
  ax1.set_ylabel("Xc")
  ax2.set_ylabel("Yc")
  ax3.set_ylabel("Time Detrended Flux")
  ax3.set_xlabel("BJD (Days)")
  pl.tight_layout()
  pl.savefig(outputfolder+"/ThrusterEvent.png")
  pl.close('all')

  #Remove the thruster events
  time = time[~ThrusterIndices]
  flux = flux[~ThrusterIndices]
  TimeDetrendedFlux = TimeDetrendedFlux[~ThrusterIndices]
  TargetArc = TargetArc[~ThrusterIndices]

  #Divide the data into different chunks
  time_chunks = list(chunks(time,chunksize))
  flux_chunks = list(chunks(flux,chunksize))
  detrendedflux_chunks = list(chunks(TimeDetrendedFlux,chunksize))
  Arc_chunks = list(chunks(TargetArc,chunksize))

  #if the last chunk is really small
  if len(time_chunks[-1])<chunksize/2.0:
      time_chunks[-2].extend(time_chunks[-1])
      time_chunks.pop()
      flux_chunks[-2].extend(flux_chunks[-1])
      flux_chunks.pop()
      detrendedflux_chunks[-2].extend(detrendedflux_chunks[-1])
      detrendedflux_chunks.pop()
      Arc_chunks[-2].extend(Arc_chunks[-1])
      Arc_chunks.pop()


  PositionTrend = []
  GoodTime = []
  GoodFlux = []
  ArcFitArray  = []
  ArcFitArrayPlot = []

  for t,f,df,Arc  in zip(time_chunks,flux_chunks,detrendedflux_chunks,Arc_chunks):
      #Find the arclength vs flux relationship
      ChunkSize = chunksize/3.0 #50
      temp_A = np.copy(Arc)
      temp_df = np.copy(df)
      Index = np.argsort(temp_A)
      temp_A = temp_A[Index]
      temp_df = temp_df[Index]
      Success = 0
      for i in range(3):
          try:
              N = int(len(temp_A)/ChunkSize)
              Location = [int((i+0.5)*ChunkSize) for i in range(N)]
              knots = [temp_A[i] for i in Location]
              spl_arc = spline(temp_A, temp_df, knots, k=2)
              Detrended_Pos = spl_arc(temp_A)
              Residual = np.abs(temp_df - Detrended_Pos)
              Indices = Residual<2.0*np.std(Residual)
              temp_A = temp_A[Indices]
              temp_df = temp_df[Indices]
              Success=1
          except:
              '''This avoid bad fitting data that does not satisfy Schoenbery Whitney
              condition, which is required to be fullfilled for spline fitting'''
              pass
      if Success==1:
          Detrended_Pos = spl_arc(Arc)
          PositionTrend.extend(Detrended_Pos)

          PlotArc = np.copy(Arc)
          Plot_df = np.copy(Detrended_Pos)

          Index = np.argsort(PlotArc)
          PlotArc = PlotArc[Index]
          Plot_df = Plot_df[Index]

          ArcFitArray.append([temp_A,temp_df])
          ArcFitArrayPlot.append([PlotArc,Plot_df])
          GoodTime.extend(t)
          GoodFlux.extend(f)


  pl.figure()
  pl.plot(TargetArc,TimeDetrendedFlux,"ro")
  for i in range(len(ArcFitArray)):
     pl.plot(ArcFitArray[i][0],ArcFitArray[i][1],"ko")
  for i in range(len(ArcFitArrayPlot)):
     pl.plot(ArcFitArrayPlot[i][0], ArcFitArrayPlot[i][1], "g-", lw=3 )
  Std = np.std(TimeDetrendedFlux)
  pl.ylim([1-3*Std, 1+3*Std])
  pl.savefig(outputfolder+"/Arc_vs_Flux.png")

  PositionTrend = np.array(PositionTrend)
  PosDetrendedLC = GoodFlux/PositionTrend

  #Spline fitting
  TempFlux = np.copy(PosDetrendedLC)
  TempTime = np.copy(GoodTime)
  ChunkSize = 50 #chunk every 1.5 day

  for k in range(3):
      N = int(len(TempTime)/ChunkSize)
      Location = [int((i+0.5)*ChunkSize) for i in range(N)]
      knots = [TempTime[i] for i in Location]
      spl = spline(TempTime, TempFlux, knots, k=2)
      DetrendedFlux = spl(TempTime)
      Residual = np.abs(TempFlux -DetrendedFlux)
      Indices = Residual<3.0*np.std(Residual)
      TempFlux = TempFlux[Indices]
      TempTime = TempTime[Indices]
  TimeDetrend = spl(GoodTime)
  PosTimeDetrendedLC = np.array(PosDetrendedLC)/TimeDetrend


  #Add a second step of flattening
  #Spline fit the time until convergence
  TempFlux = np.copy(PosTimeDetrendedLC)
  TempTime = np.copy(GoodTime)


  for i in range(3):
      _, var = moving_average(TempFlux)
      spl =  UnivariateSpline(TempTime, TempFlux, w=1/np.sqrt(var))
      SplEstimatedFlux = spl(TempTime)
      Residual = np.abs(TempFlux - SplEstimatedFlux)
      Indices = Residual<2.0*np.std(Residual)
      TempFlux = TempFlux[Indices]
      TempTime = TempTime[Indices]


  TimeDetrendNew = spl(GoodTime)
  PosTimeDetrendedLC = np.array(PosTimeDetrendedLC)/TimeDetrendNew

  np.savetxt(os.path.join(outputfolder, 'PositionDetrended.csv'),np.transpose([GoodTime, PosDetrendedLC]),header='Time, Flux', delimiter=',')

  pl.figure(figsize=(15,8))
  pl.subplot(311)
  pl.plot(time, flux, "go", markersize=1.5, label="Raw Flux")
  pl.legend()
  pl.subplot(312)
  pl.plot(GoodTime, PosDetrendedLC,"bo", markersize=1.5, label="Position Detrended Flux")
  pl.plot(GoodTime,TimeDetrend,"r-", lw=2, label="Flattening" )
  pl.legend()
  pl.subplot(313)
  pl.plot(GoodTime, PosTimeDetrendedLC, "go", markersize=1.5, label="Final Flattened Light Curve")
  pl.legend()
  pl.savefig(outputfolder+"/Detrending.png")

  return [GoodTime,PosTimeDetrendedLC]
