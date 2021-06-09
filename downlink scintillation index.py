#calculation of scintillation index

from __future__ import division
import math
import mpmath
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

#===================== scintillation loss paper ==========================

#Speqtre Qubesat quantum  
w = 4.5 #nominal
A = 1.7e-15 #nominal
h_0 = 11
h_satellite = 500e3 #13.5e6
wavelength = 780e-9
D = 0.6

#OICETS  
#w = 10
#A = 1.7e-14
#h_0 = 600
#h_satellite = 610e3
#wavelength = 847e-9
#D = 0.4

#OPALS
#w = 10
#A = 1e-13
#h_0 = 600
#h_satellite = ??
#wavelength - 1550e-9


Hd = 12e3
emax = 10
k = 2*math.pi/wavelength



def Cn_Square(h):
    #return 0.00594*((w/27)**2)*((h*1e-5)**10)*math.exp(-h/1000)\
    #            +2.7e-16*math.exp(-h/1500) \
    #             + A * math.exp(-h/100)

    return 0.00594*((w/21)**2)*((h*1e-5)**10)*math.exp(-h/1000)\
                +2.7e-16*math.exp(-h/1500) \
                 + A * math.exp(-h/700)*math.exp(-(h-h_0)/100)


def PSI(e):
    ZenithAngle = mpmath.sec(math.radians(90 - e))

    g = lambda h: Cn_Square(h) * (h-h_0)**(5/6) 
    rytov = 0.225*k**(7/6)*ZenithAngle**(11/6)*quad(g, h_0, h_satellite)[0]

    L = Hd * (e/90) / ((e/90)**2 + (emax/90)**2)
    p = 1.5 * math.sqrt((L*wavelength)/(2*math.pi))
    af1 = (D/(2*p))**2
    af = (1 + 1.062 * af1)**(-7/6)
    PSI = rytov * af
    
    return rytov,PSI


x = np.linspace(1,90,1000)
y = np.array([PSI(e)[0] for e in x])
plt.plot(x,y,'b-')
yy = np.array([PSI(e)[1] for e in x])
plt.plot(x,yy,'r-')

plt.ylabel('Scintillation Index')
plt.yscale('log')
plt.xlabel('Link-Elevation in degree')
plt.legend(["SI", "PSI"])
#plt.show()







#=========== larry andrews ch12 - aperture averaging (pg496) ====================


def complex_quad(func, a, b):
    def real_func(x):
        return np.real(func(x))
    def imag_func(x):
        return np.imag(func(x))
    real_integral = quad(real_func, a, b)
    imag_integral = quad(imag_func, a, b)
    return (real_integral[0]+ 1j*imag_integral[0])

def SI(D):
    ZenithAngle = 60
    L = (h_satellite - h_0)/math.cos(math.radians(ZenithAngle))
  
    ZenithAngle = mpmath.sec(math.radians(ZenithAngle))
    a = 8.70*k**(7/6)*(h_satellite-h_0)**(5/6)*ZenithAngle**(11/6)
    f = lambda h: Cn_Square(h) * ((k*D**2/(16*L)+1j*((h-h_0)/(h_satellite-h_0)))**(5/6) - (k*D**2/(16*L))**(5/6))
    b = np.real(complex_quad(f,h_0,h_satellite))
    return a*b,L


x = np.linspace(0,60,1000)
y = np.array([SI(D) for D in x*1e-2])
plt.plot(x,y)
plt.xlabel('Aperture Diameter (cm)')
plt.ylabel('aperture averaged SI')
plt.show()
