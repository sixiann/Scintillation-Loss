#fade statistics - larry c andrews ch12

from __future__ import division
import math
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt


I_mean = 1
W_0 = 0.267
SI = 0.32977623493084773

#unbounded plane wave,downlink
theta = 1
lambda_ = 0
W = W_0/(math.sqrt(theta**2+lambda_**2))
#W_LT = W
W_LT = 3.1


#===================================== lognormal graph =====================

def lognormpdf(I): 
    pdf1 = 1/(I*math.sqrt(SI*2*math.pi))
    pdf2 = ((math.log(I/I_mean)+0.5*SI)**2)/(2*SI)
    pdf = pdf1 * math.exp(-pdf2)
    return pdf

x = np.linspace(0.1,2.5,100)
y = np.array([lognormpdf(I) for I in x])
plt.plot(x,y)
plt.ylabel('PDF')
plt.xlabel('Intensity')
plt.title('Lognormal PDF of received intensity')
plt.show()


#====================================== fractional fade time =================


def p_fade(Ft): 
    I_thr = I_mean / (10 ** (0.1*Ft))
    g = lambda I: lognormpdf(I)
    p_fade = quad(lognormpdf,0,I_thr)[0]

    return p_fade

x = np.linspace(0,3,1000)
y0 = np.array([p_fade(Ft) for Ft in x])
plt.plot(x,y0)
plt.yscale('log')
plt.ylabel('Probability of Fade')
plt.xlabel('Fade threshold, Ft (dB)')

plt.show()

#============================ expected number of fades =======================

def n_fade(Ft):
    I_thr = I_mean / (10 ** (0.1*Ft))
    v_0 = 550
    nfade = v_0 * math.exp(-(0.5*SI - math.log(((W_LT/W_0)**2)*I_mean)-0.23*Ft)**2 / (2*SI))

    return nfade

y1 = np.array([n_fade(Ft) for Ft in x])
plt.plot(x,y1)
#plt.yscale('log')
plt.ylabel('Expected No. of Fades/Second')
plt.xlabel('Fade threshold, Ft (dB)')

plt.show()

#=============================== mean fade time ====================================

y3 = np.array([p_fade(Ft)/n_fade(Ft) for Ft in x])
plt.plot(x,y3)
#plt.yscale('log')
plt.ylabel('Mean Duration of Fade (s)')
plt.xlabel('Fade threshold, Ft (dB)')


plt.show()
