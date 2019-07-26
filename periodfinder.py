'''
#
# Module to find periodicities of a light curve of flux using BLS. Best periods are automatically folded, other periods are tried and other nice stuff can be done.
# Credit for this Box-least-square (BLS) algorithm goes to Ruth Angus and Dan Foreman-Mackey! See https://github.com/dfm/python-bls, but included here for completeness
#
#
'''

# general python files
from __future__ import division
import os
import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import LSQUnivariateSpline as spline
from auxiliaries import *

import bls

def fold_data(t,y,period):
  # simple module to fold data based on period

  folded = t % period
  inds = np.array(folded).argsort()
  t_folded = folded[inds]
  y_folded = y[inds]

  return t_folded,y_folded


def get_period(t,f_t,outputpath='',starname=''):

  # here we use a BLS algorithm to create a periodogram and find the best periods. The BLS is implemented in Python by Ruth Angus and Dan Foreman-Mackey

  outputfolder = os.path.join(outputpath,str(starname))
  fmin = 2.0/((t[len(t)-1]-t[0])) # minimum frequency. we can't find anything longer than 90 days obviously
  nf = 3e6 # amount of frequencies to try
  df = 1e-5#0.00001 # frequency step

  qmi = 0.0005 # min relative length of transit (in phase unit)
  qma = 0.1 # max relative length of transit (in phase unit)
  nb = 200 # number of bins in folded LC

  u = np.linspace(fmin,fmin + nf*df,nf)
  v = np.array(0)
  t = np.array(t)
  f_t = np.array(f_t)

  t_orig = np.copy(t)
  f_t_orig = f_t

  results = bls.eebls(t,f_t,t,f_t,nf,fmin,df,nb,qmi,qma)
  freqlist = u
  powers = results[0]
  period = results[1]
  #print(results[2], results[3])
  depth = results[2]
  duration = results[3]

  folded,f_t_folded = fold_data(t,f_t,period)

  try:
    np.savetxt(os.path.join(outputfolder, 'Flattened.txt'),np.transpose([t, f_t]),header='Time, Flux')
    np.savetxt(os.path.join(outputfolder, 'PhaseFolded.txt'),np.transpose([folded,f_t_folded]),header='Time, Flux')
    np.savetxt(os.path.join(outputfolder, 'OptimalDepth.txt'),[depth],header='Depth')
    np.savetxt(os.path.join(outputfolder, 'OptimalDuration.txt'),[duration],header='Fractional Duration')
  except:
    os.mkdir(outputfolder)
    np.savetxt(os.path.join(outputfolder, 'Flattened.txt'),np.transpose([t, f_t]),header='Time, Flux')
    np.savetxt(os.path.join(outputfolder, 'PhaseFolded.txt'),np.transpose([folded,f_t_folded]),header='Time, Flux')
    np.savetxt(os.path.join(outputfolder, 'OptimalDepth.txt'),[depth],header='Depth')
    np.savetxt(os.path.join(outputfolder, 'OptimalDuration.txt'),[duration],header='Fractional Duration')

  t_foldbin,f_t_foldbin,stdv_foldbin = rebin_dataset(folded,f_t_folded,15)
  f_t_smooth = savitzky_golay(f_t_folded,101,1)

  # unravel again
  n_start = int(np.round(t[0] / period))
  n_end = np.round(t[-1] / period) + 1
  i = n_start

  t_unravel = []
  f_t_unravel = []
  while i < n_end:
    t_unravel.append(np.array(folded) + i*period + t_orig[0])
    f_t_unravel.append(np.array(f_t_smooth))

    #pl.plot(t_unravel[i],f_t_unravel[i],color='black',lw='1.5')
    i = i + 1

  if abs(period-0.24)<0.01:
    freqlist,powers = maskout_freqs(freqlist,powers,freq=0.245) # spacecraft freqs
    Location = np.where(powers==max(powers))
    period = freqlist[Location[0]][0]
  print("The best period is: ", period)
  return folded,f_t_folded,period,freqlist,powers


def maskout_freqs(freqs,power,freq=float,lim=0.005):
  # little definition to remove certain frequencies and their nearby surroundings from an array, so that other frequencies can be found which are truly different
  freqs = np.array(freqs)
  power = np.array(power)
  selection = ~((freqs > (freq - lim))*((freqs < (freq + lim))))
  newfreqs = freqs[selection]
  newpower = power[selection]

  return newfreqs,newpower


def make_combo_figure(filepath, t,f_t,period,freqs,power,starname='',outputpath=''):
  #
  # This definition can be used to make a single overview figure, showing the lightcurve + a zoom, a BLS periodogram, and folded (+ smoothed) light curves based on the best frequencies and their multiples
  # This figure is used to eyeball good candidates
  #
  outputfolder = os.path.join(outputpath,str(starname))
  t = np.array(t)
  f_t = np.array(f_t)
  freqs = np.array(freqs)
  power = np.array(power)

  plt.rc('xtick', labelsize = 15)
  plt.rc('ytick', labelsize = 15)

  plt.figure('Combo figure',figsize=(35.,25.))
  ax1 = plt.subplot2grid((7,3), (0,0), colspan=3,rowspan=2)
  ax2 = plt.subplot2grid((7,3), (2,0), colspan=3,rowspan=2)
  ax3 = plt.subplot2grid((7,3), (4,0), colspan=3)
  ax4 = plt.subplot2grid((7,3), (5,0))#, rowspan=2)
  ax5 = plt.subplot2grid((7,3), (5,1))
  ax6 = plt.subplot2grid((7,3), (5,2))
  ax7 = plt.subplot2grid((7,3), (6,0))#, rowspan=2)
  ax8 = plt.subplot2grid((7,3), (6,1))
  ax9 = plt.subplot2grid((7,3), (6,2))

  sn = np.max(power)/np.median(power)

  if sn > 3.:
    titlecolor = 'green'
    outputfigfolder = os.path.join(outputpath,'figs/high_sn/')
  else:
    titlecolor = 'red'
    outputfigfolder = os.path.join(outputpath,'figs/low_sn/')
  plt.suptitle('star = ' + str(starname),fontsize=35,color=titlecolor)
  ax1.plot(t,f_t,'.-')

  ax2.plot(t,f_t-1,'.-.',lw=0.5)
  ax2.set_ylim([-0.002,0.002])
  ax2.set_xlabel('Time [d]',fontsize=20)
  #ax1.xaxis.set_label_position('top')

  P_min = freqs[-1]
  P_max = freqs[0]
  freq_best = 1./period

  #saving in the summary
  #RecordFile = open(outputpath+"/RunSummary.csv","a")
  #RecordFile.write(str('%.2f' %sn) +','+str('%.5f' %freq_best)+',')
  #RecordFile.close()

  label = ' (Pmin = ' + str(np.round(P_min,3)) + ', Pmax = ' + str(np.round(P_max,3)) + ')'
  freqs = np.array(freqs)
  freqs = 1./freqs
  ax3.plot(freqs,power,'-',lw=2,label=label,color='black',zorder = 4)
  ax3.axvline(period,lw=1,color='red',label='Period = ' + str(np.round(period,3)),zorder=3)
  ax3.legend(loc='best',fontsize=20)

  t_folded,f_t_folded = fold_data(t,f_t,period)
  f_t_smooth = savitzky_golay(f_t_folded,51,1)
  ax4.plot(t_folded,f_t_folded,'.',color='grey',label='Period = ' + str(np.round(period,3))+ ' S/N=' + str(np.round(sn,2)))
  ax4.plot(t_folded,f_t_smooth,color='red',lw=3)
  ax4.legend(loc='best',fontsize=20)

  t_folded_2P,f_t_folded_2P = fold_data(t,f_t,period*2.)
  f_t_smooth_2P = savitzky_golay(f_t_folded_2P,51,1)
  ax5.plot(t_folded_2P,f_t_folded_2P,'.',color='grey',label='2 x Period = ' + str(np.round(period*2.,3)))
  ax5.plot(t_folded_2P,f_t_smooth_2P,color='red',lw=3)
  ax5.legend(loc='best',fontsize=20)

  t_folded_halfP,f_t_folded_halfP = fold_data(t,f_t,period/2.)
  f_t_smooth_halfP = savitzky_golay(f_t_folded_halfP,51,1)
  ax6.plot(t_folded_halfP,f_t_folded_halfP,'.',color='grey',label='Period/2 = ' + str(np.round(period/2.,3)))
  ax6.plot(t_folded_halfP,f_t_smooth_halfP,color='red',lw=3)
  ax6.legend(loc='best',fontsize=20)


  # find next best period, avoid the one already taken, and always avoid 0.5 and 0.25 days
  newfreqs,newpower = maskout_freqs(freqs,power,freq=period)
  newfreqs,newpower = maskout_freqs(newfreqs,newpower,freq=period/2.)
  newfreqs,newpower = maskout_freqs(newfreqs,newpower,freq=period*2.)
  newfreqs,newpower = maskout_freqs(newfreqs,newpower,freq=period*3.)
  newfreqs,newpower = maskout_freqs(newfreqs,newpower,freq=period/3.)

  newfreqs,newpower = maskout_freqs(newfreqs,newpower,freq=0.25) # spacecraft freqs
  newfreqs,newpower = maskout_freqs(newfreqs,newpower,freq=0.24) # spacecraft freqs
  newfreqs,newpower = maskout_freqs(newfreqs,newpower,freq=0.5) # spacecraft freqs
  newfreqs,newpower = maskout_freqs(newfreqs,newpower,freq=0.125) # spacecraft freqs

  newbestfreq2 = newfreqs[np.argmax(newpower)]
  period2 = newbestfreq2
  sn2 = np.max(newpower)/np.median(newpower)

  newfreqs,newpower = maskout_freqs(newfreqs,newpower,freq=newbestfreq2)
  newfreqs,newpower = maskout_freqs(newfreqs,newpower,freq=newbestfreq2/2.)
  newfreqs,newpower = maskout_freqs(newfreqs,newpower,freq=newbestfreq2*2.)
  newbestfreq3 = newfreqs[np.argmax(newpower)]
  period3 = newbestfreq3
  sn3 = np.max(newpower)/np.median(newpower)

  newfreqs,newpower = maskout_freqs(newfreqs,newpower,freq=newbestfreq3)
  newfreqs,newpower = maskout_freqs(newfreqs,newpower,freq=newbestfreq3/2.)
  newfreqs,newpower = maskout_freqs(newfreqs,newpower,freq=newbestfreq3*2.)
  newbestfreq4 = newfreqs[np.argmax(newpower)]
  period4 = newbestfreq4
  sn4 = np.max(newpower)/np.median(newpower)

  ax3.axvline(newbestfreq2,lw=1,color='blue',label='Period2 = ' + str(np.round(period2,3)),ls='--',zorder=2)
  ax3.axvline(newbestfreq3,lw=1,color='orange',label='Period3 = ' + str(np.round(period3,3)),ls='--',zorder=1)
  ax3.axvline(newbestfreq4,lw=1,color='green',label='Period4 = ' + str(np.round(period4,3)),ls='--',zorder=0)
  ax3.legend(loc='best',ncol=5,fontsize = 20)

  t_folded,f_t_folded = fold_data(t,f_t,period2)
  f_t_smooth = savitzky_golay(f_t_folded,51,1)
  ax7.plot(t_folded,f_t_folded,'.',color='grey',label='Period = ' + str(np.round(period2,3))+ ' S/N=' + str(np.round(sn2,2)))
  ax7.plot(t_folded,f_t_smooth,color='blue',lw=3)
  ax7.legend(loc='best',fontsize = 20)

  t_folded,f_t_folded = fold_data(t,f_t,period3)
  f_t_smooth = savitzky_golay(f_t_folded,51,1)
  ax8.plot(t_folded,f_t_folded,'.',color='grey',label='Period = ' + str(np.round(period3,3))+ ' S/N=' + str(np.round(sn3,2)))
  ax8.plot(t_folded,f_t_smooth,color='orange',lw=3)
  ax8.legend(loc='best',fontsize = 20)

  t_folded,f_t_folded = fold_data(t,f_t,period4)
  f_t_smooth = savitzky_golay(f_t_folded,51,1)
  ax9.plot(t_folded,f_t_folded,'.',color='grey',label='Period = ' + str(np.round(period4,3))+ ' S/N=' + str(np.round(sn4,2)))
  ax9.plot(t_folded,f_t_smooth,color='green',lw=3)
  ax9.legend(loc='best',fontsize=20)


  ax7.set_xlabel('Time [d]',fontsize=20)
  ax8.set_xlabel('Time [d]',fontsize=20)
  ax9.set_xlabel('Time [d]',fontsize=20)
  ax3.set_ylabel('Power',fontsize=20)
  ax3.set_xlabel('Period [d]',fontsize=20)
  ax1.set_ylabel('Flux',fontsize=20)
  ax2.set_ylabel('Flux (zoom)',fontsize=20)
  ax4.set_ylabel('Flux',fontsize=20)
  ax7.set_ylabel('Flux',fontsize=20)
  if not os.path.exists(outputfigfolder):
    os.makedirs(outputfigfolder)

  plt.tight_layout()
  plt.subplots_adjust(top=0.95)
  #Saving the figure
  plt.savefig(os.path.join(outputfigfolder, 'combo_' + 'star_' + str(starname) + '.png'),figsize=(10.,20.))
  plt.savefig(os.path.join(outputfolder, 'BLS_Search.png'),figsize=(10.,20.))
  #plt.show()
  plt.close('all')
