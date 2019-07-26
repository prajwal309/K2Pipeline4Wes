from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

Location = "TestStars/ktwo211945201-c16_lpd-targ.fits"

FitsFile = fits.open(Location, memmap=True)
print(FitsFile[1].columns)

Time = FitsFile[1].data["TIME"]
Flux = FitsFile[1].data["FLUX"]
MeanFlux = np.nanmean(Flux, axis=0)
print(np.shape(MeanFlux))

plt.figure()
#plt.plot(Time, Flux, "ko")
plt.imshow(MeanFlux)
plt.show()
