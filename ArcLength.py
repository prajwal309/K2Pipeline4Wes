import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

params =   [2.05546878e+00,6.46192371e+00,3.41070676e+00,-3.07124652e-01, 1.47286775e-01, 1.55798198e-03]

#params =  np.array([0,0,0,0,2,0]) #The last value is the intercept - a_0
Mult = np.arange(len(params)-1,0,-1)

x =np.linspace(-0.5,0.5,100)
y = np.polyval(params,x)


NewParams = params[:-1]*Mult

def Integrand(x,params):
  return np.sqrt(1+(np.polyval(params,x))**2)

#for XVal in x:

ArcLength  = []
for XVal in x:
    tempArc,_ = quad(Integrand,0,XVal,args=(NewParams),epsabs=1e-5)
    ArcLength.append(tempArc)



print ArcLength
print "Is it working AT ALL?"
plt.figure()
plt.plot(x,ArcLength, 'k.')
plt.plot(x,y,'b.')
#plt.axis('equal')
plt.show()
