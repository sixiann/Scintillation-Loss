from __future__ import division
import math
import numpy as np
import mpmath
from scipy.integrate import quad
from scipy import special
import matplotlib.pyplot as plt




def complex_quad(func, a, b):
    def real_func(x):
        return np.real(func(x))
    def imag_func(x):
        return np.imag(func(x))
    real_integral = quad(real_func, a, b)
    imag_integral = quad(imag_func, a, b)
    return (real_integral[0]+ 1j*imag_integral[0])


def Cn_Square(h):
    w = 21
    A = 1.7e-14
    return 0.00594*((w/27)**2)*((h*1e-5)**10)*np.exp(-h/1000)\
            +2.7e-16*np.exp(-h/1500) \
                + A * np.exp(-h/100)

#pg 490
def xi(h,h_0,h_satellite):
    uplink = 1 - ((h-h_0)/(h_satellite-h_0))
    downlink = (h-h_0)/(h_satellite-h_0)
    return uplink,downlink


#pg 488-489
def GaussParam(h_0,h_satellite,ZenithAngle,wavelength,W_0):
    L = (h_satellite - h_0)/math.cos(math.radians(ZenithAngle))
    k = 2*math.pi/wavelength

    #theta0 = 1 - L/F_0
    theta0 = 1 #collimated beam
    lambda0 = 2*L/(k*W_0)**2

    theta = theta0/(theta0**2 + lambda0**2)
    thetabar = 1 - theta
    lambda_= lambda0/(theta0**2 + lambda0**2)

    #plane wave
   # theta = 0
   # lambda_ = 0

    return theta0, theta, thetabar, lambda0, lambda_




# (53) pointing error variance  
def pointerror(h_0,h_satellite,ZenithAngle,wavelength,W_0,kr,r_0,n_int=1000):

    ZA = mpmath.sec(math.radians(ZenithAngle))
    L = (h_satellite - h_0)/math.cos(math.radians(ZenithAngle))

    h = np.linspace(h_0,h_satellite,n_int,axis=-1)
    integ = np.trapz(Cn_Square(h)*(xi(h,h_0,h_satellite)[0])**2,h,axis=-1)
    s_pe = 7.25*(h_satellite-h_0)**2*ZA**3*W_0**(-1/3) \
       * (1 - ((kr*W_0)**2/(1+(kr*W_0)**2))**(1/6)) * integ
    
    #f = lambda h: Cn_Square(h) * (xi(h,h_0,h_satellite)[0])**2
    #s_pe = 7.25*(h_satellite-h_0)**2*ZA**3*W_0**(-1/3) \
    #   * (1 - ((kr*W_0)**2/(1+(kr*W_0)**2))**(1/6)) * quad(f,h_0,h_satellite)[0]
    #r_0 = 19e-2
    
    
    Cr = 2*math.pi
    b = (Cr*W_0/r_0)**2
    #s_pe = 0.54 * (h_satellite-h_0)**2 * ZA**2 * (wavelength/2*W_0)**2 * (2*W_0/r_0)**(5/3)* (1-(b/(1+b))**(1/6))
    
    a_pe = math.sqrt(s_pe)/L 

    return s_pe, a_pe



#pg 489, 500
def waist(h_0,h_satellite,ZenithAngle,wavelength,W_0,r_0):
   
    theta0 = GaussParam(h_0,h_satellite,ZenithAngle,wavelength,W_0)[0]
    lambda0 = GaussParam(h_0,h_satellite,ZenithAngle,wavelength,W_0)[3]
    lambda_ = GaussParam(h_0,h_satellite,ZenithAngle,wavelength,W_0)[4]
    ZA = mpmath.sec(math.radians(ZenithAngle))
    k = 2*math.pi/wavelength
    L = (h_satellite - h_0)/math.cos(math.radians(ZenithAngle))
   
    W = W_0*math.sqrt(theta0**2 + lambda0**2)

    #WEAK
    #fg = lambda h: Cn_Square(h)*((h-h_0)/(h_satellite-h_0))**(5/3)
    #mu2d = quad(fg,h_0,h_satellite)[0]
    #W_LT = W * math.sqrt(1 + 4.35*mu2d*lambda_**(5/6)*k**(7/6)*(h_satellite-h_0)**(5/6)*ZA**(11/6))

    #STRONG
    D_0 = math.sqrt(8*W_0**2)
    if (D_0/r_0 < 1):
        W_LT = W * math.sqrt(1+(D_0/r_0)**(5/3))
    else:
        W_LT = W * (1+(D_0/r_0)**(5/3))**(3/5)

    r = math.sqrt(abs(0.5*(W_LT**2 - math.log((W_LT/W_0)**2))))
    a_r = r/L

    return W, W_LT,r,a_r


#===================================12.6.4 strong fluctuation theory ========================

                
#(62) SI of spherical wave under weak irradiance fluctuations (fig 12.18)
def sigmabu(h_0,h_satellite,ZenithAngle,wavelength,W_0,n_int=1000):
    lambda_ = GaussParam(h_0,h_satellite,ZenithAngle,wavelength,W_0)[4]
    thetabar = GaussParam(h_0,h_satellite,ZenithAngle,wavelength,W_0)[2]
    k = 2*math.pi/wavelength
    ZA = mpmath.sec(math.radians(ZenithAngle))


    h = np.linspace(h_0,h_satellite,n_int,axis=-1)
    integ = np.trapz(Cn_Square(h)*(xi(h,h_0,h_satellite)[0] * xi(h,h_0,h_satellite)[1])**(5/6),h,axis=-1)
    s_bu = 2.25*k**(7/6)*(h_satellite-h_0)**(5/6)*ZA**(11/6)*integ

    
    g = lambda h: Cn_Square(h) * (xi(h,h_0,h_satellite)[0] * xi(h,h_0,h_satellite)[1])**(5/6)
    s_bu = 2.25*k**(7/6)*(h_satellite-h_0)**(5/6)*ZA**(11/6)*quad(g,h_0,h_satellite)[0]
    
    return s_bu

print('SI of spherical wave under weak irradiance fluctuations')
print(sigmabu(11,500e3,60,915e-9,0.267,n_int=1000))

x = np.linspace(0.1,75,1000)
y = np.array([sigmabu(0,300e3,za,1.06e-6,10e-2) for za in x])
plt.plot(x,y,'r-')


#(60) tracked longitudinal strong fluctuation theory (fig 12.18)
def SIstrongtrackedL(h_0,h_satellite,ZenithAngle,wavelength,W_0):
    s_bu = sigmabu(h_0,h_satellite,ZenithAngle,wavelength,W_0)
    theta = GaussParam(h_0,h_satellite,ZenithAngle,wavelength,W_0)[1]
    a = 0.49*s_bu/((1+(1+theta)*0.56*s_bu**(6/5))**(7/6))
    b = 0.51*s_bu/((1+0.69*s_bu**(6/5))**(5/6))
    SI_L_tracked = math.exp(a+b)-1

    return SI_L_tracked

y = np.array([SIstrongtrackedL(0,300e3,za,1.06e-6,10e-2) for za in x])
plt.plot(x,y,'g-')
plt.xlabel('Zenith Angle (degrees)')
plt.ylabel('Scintillation index')
plt.legend(['weak fluctuation theory','strong fluctuation theory'])
plt.title('on-axis, tracked')
#plt.show()

print ('tracked longitudinal strong fluctuation theory')
print(SIstrongtrackedL(11,500e3,60,915e-9,0.267))

#(61) untracked radial strong fluctuation theory
def SIstronguntrackedR(h_0,h_satellite,ZenithAngle,wavelength, W_0,r_0,kr,r):
    L = (h_satellite - h_0)/math.cos(math.radians(ZenithAngle))
    a_r = r/L
    a_pe = pointerror(h_0,h_satellite,ZenithAngle,wavelength,W_0,kr,r_0)[1]
    W = waist(h_0,h_satellite,ZenithAngle,wavelength,W_0,r_0)[0]
    ZA = mpmath.sec(math.radians(ZenithAngle))
    SI_L_tracked = SIstrongtrackedL(h_0,h_satellite,ZenithAngle,wavelength,W_0)

    a = 5.95 * (h_satellite-h_0)**2 * ZA**2 * (2*W_0/r_0)**(5/3)
    SI_r_untracked = a * (((a_r-a_pe)/W)**2 * ((a_r-a_pe)>0) + (a_pe/W)**2) + SI_L_tracked
    
    return SI_r_untracked



#fig 12.15
x = np.linspace(0,6,1000)
y = np.array([SIstronguntrackedR(0,300e3,0,1.6e-6,5e-2,20e-2,math.pi/(20e-2),r) for r in x])
plt.plot(x,y)
plt.ylabel('SI')
plt.yscale('log')
plt.xlabel('Radial distance r (m)')
plt.ylabel('Scintillation Index')
plt.show()






# ================================12. 6. 3 ===========================================




#(55) mu3u
def mu3uparam(h_0,h_satellite,ZenithAngle,wavelength,W_0):
    #def xi(h):
    #   return 1 - (h-h_0)/(h_satellite-h_0)
    lambda_ = GaussParam(h_0,h_satellite,ZenithAngle,wavelength,W_0)[4]
    thetabar = GaussParam(h_0,h_satellite,ZenithAngle,wavelength,W_0)[2]
    
    fg = lambda h: Cn_Square(h) * ((xi(h,h_0,h_satellite)[0]* (lambda_*xi(h,h_0,h_satellite)[0] + 1j*(1-thetabar*xi(h,h_0,h_satellite)[0])))**(5/6) - lambda_**(5/6)*xi(h,h_0,h_satellite)[0]**(5/3))
    mu3u = np.real(complex_quad(fg,h_0,h_satellite))



    return mu3u


#(58) tracked longitudinal 
def sigmabuweak(h_0,h_satellite,ZenithAngle,wavelength,W_0,n_int=1000):
    def xi(h):
        return 1 - (h-h_0)/(h_satellite-h_0)
    k = 2*math.pi/wavelength
    ZA = mpmath.sec(math.radians(ZenithAngle))

    mu3u = mu3uparam(h_0,h_satellite,ZenithAngle,wavelength,W_0)
    #s_bu = 8.70*mu3u*k**(7/6)*(h_satellite-h_0)**(5/6)*ZA**(11/6)


    #pg 527 q11
    #f = lambda h: Cn_Square(h) * (h-h_0)**(5/6) * xi(h)**(5/6)
    #s_bu = 2.25*k**(7/6)*ZA**(11/6)*quad(f, h_0, h_satellite)[0]


    h = np.linspace(h_0,h_satellite,n_int,axis=-1)
    integ = np.trapz(Cn_Square(h) * (h-h_0)**(5/6) * xi(h)**(5/6),h,axis=-1)
    s_bu = 2.25*k**(7/6)*ZA**(11/6)*integ
    
    return s_bu


#(54) untracked longitudinal
def SIweakuntrackedL(h_0,h_satellite,ZenithAngle,wavelength,W_0,r_0,kr):
    a_pe = pointerror(h_0,h_satellite,ZenithAngle,wavelength,W_0,kr,r_0)[1]
    W = waist(h_0,h_satellite,ZenithAngle,wavelength,W_0,r_0)[0]
    ZA = mpmath.sec(math.radians(ZenithAngle))
    mu3u = mu3uparam(h_0,h_satellite,ZenithAngle,wavelength,W_0)
    k = 2*math.pi/wavelength
    
    L = (h_satellite - h_0)/math.cos(math.radians(ZenithAngle))
    a_pe = 30e-2/L

    SI_L_untracked = 5.95 * (h_satellite-h_0)**2 * ZA**2 * (2*W_0/r_0)**(5/3) \
                 * (a_pe/W)**2 + 8.70*mu3u*k**(7/6)*(h_satellite-h_0)**(5/6)*ZA**(11/6)
    
    return SI_L_untracked


#fig 12.12
x = np.linspace(1e3,1e5,500)
y = np.array([SIweakuntrackedL(0,h,0,1.55e-6,10e-2,19e-2,40) for h in x])
y2 = np.array([sigmabuweak(0,h,0,1.55e-6,10e-2) for h in x])
plt.plot(x,y,'r-')
#plt.plot(x,y2,'g-')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Altitude(m)')



def rytov(h_0,h_satellite,ZenithAngle,wavelength,n_int=1000):
    ZA = mpmath.sec(math.radians(ZenithAngle))
    k = 2*math.pi/wavelength
    #g = lambda h: Cn_Square(h) * (h-h_0)**(5/6) 
    #rytov = 0.225*k**(7/6)*ZA**(11/6)*quad(g, h_0, h_satellite)[0]

    h = np.linspace(h_0,h_satellite,n_int,axis=-1)
    integ = np.trapz(Cn_Square(h) * (h-h_0)**(5/6),h,axis=-1)
    rytov = 0.225*k**(7/6)*ZA**(11/6)*integ
    
    return rytov

y3 = np.array([rytov(0,h,0,1.55e-6) for h in x])
plt.plot(x,y3,'b')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Atltitude(m)')
plt.show()




#(56) untracked radial
def SIweakuntrackedR(h_0,h_satellite,ZenithAngle,wavelength,W_0,r_0,kr):
    a_r = waist(h_0,h_satellite,ZenithAngle,wavelength,W_0,r_0)[3]
    a_pe = pointerror(h_0,h_satellite,ZenithAngle,wavelength,W_0,kr,r_0)[1]
    W = waist(h_0,h_satellite,ZenithAngle,wavelength,W_0,r_0)[0]
    ZA = mpmath.sec(math.radians(ZenithAngle))
    SI_L_untracked = SIweakuntrackedL(h_0,h_satellite,ZenithAngle,wavelength,W_0,r_0,kr)

    if (a_r-a_pe)>=0:
        SI_r_untracked = 5.95 * (h_satellite-h_0)**2 * ZA**2 * (2*W_0/r_0)**(5/3) \
                     * ((a_r-a_pe)/W)**2 + SI_L_untracked
    elif (a_r-a_pe<0):
        SI_r_untracked = SI_L_untracked
    return SI_r_untracked

#x2 = np.linspace(4e-2,5e-2,500)
#y = np.array([SIweakuntrackedR(0,300e3,60,0.84e-6,w,6e-2,3.86/6e-2) for w in x2])
#plt.plot(x2,y)
#plt.yscale('log')
#plt.show()




