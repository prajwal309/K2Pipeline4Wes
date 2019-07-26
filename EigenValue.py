import numpy as np
import matplotlib.pyplot as pl

params = np.random.normal(size=4)

x = np.linspace(0,3,50)
y = np.polyval(params, x)+ 0.2*np.random.normal(size=len(x))
x = x - np.median(x)
y = y - np.median(y)

N = len(x)
Centroids = np.vstack([x,y])
CoVar = np.cov(Centroids)
e1, v1 = np.linalg.eig(CoVar)


ys = np.dot(v1.T, Centroids)


pl.figure()
pl.plot(x,y, "ko")
pl.plot(ys[1,:],ys[0,:], "ro")
pl.show()
