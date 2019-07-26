import os
import glob
import imageio
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits

folder = 'Jensen18_fits'

stars = glob.glob(folder + '/*.fits')

for s in stars:
	starname = s.split('/')[-1][4:13]
	img = fits.open(s)
	flux = img[1].data['flux']

	for i in range(0, len(flux), 20):
		f = flux[i]
		plt.imshow(f, origin = 'lower')
		plt.title(str(i))
		plt.savefig(str(i) + '.png')
		plt.close()

	images = []
	for i in range(0, len(flux), 20):
		images.append(imageio.imread(str(i) + '.png'))
	imageio.mimsave(starname + '.gif', images)

	os.system('rm -r *.png')