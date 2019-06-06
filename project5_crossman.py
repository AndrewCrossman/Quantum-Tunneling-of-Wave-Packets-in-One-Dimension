# -*- coding: utf-8 -*-
"""
@Title:     Physics 660 Project Four
@Author     Andrew Crossman
@Date       May 12th, 2019
"""
###############################################################################
# IMPORTS
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
__version__ = '1.0'
print(__doc__)
print('version:', __version__)
###############################################################################
# CONSTANTS
###############################################################################
hbar = m = 1            #normailized planck constant and mass
left_edge = .6          #barrier start
V = 9.8*10**5           #barrier height
x_0 = .3                #wave center              
tot_steps = 8000        #total number of time steps
dx = .001               #x grid point spacing
dt = .0000001           #time spacing
Nx = int(1/dx)          #number of grid points 
C1 = dt/(2.0*dx**2)
C2 = dt
###############################################################################
# FUNCTION DEFINITIONS
###############################################################################
#test wave potential
def potential_one():
    v = np.zeros(Nx+2)
    return v

#single barrier potential
def potential_two(d):
    v = []
    for i in range(0,Nx+2):
        if i >= int((Nx+2)*.6) and i < int((Nx+2)*.6)+Nx*d:
            v.append(V)
        else:
            v.append(0)
    return v

#two barrier potential
def potential_three():
    v = []
    for i in range(0,Nx+2):
        if (i>=int((Nx+2)*.6) and i<int((Nx+2)*.6)+Nx/1000) or (i>=int((Nx+2)*.6)+Nx/1000*5 and i<int((Nx+2)*.6)+Nx/1000*6):
            v.append(V)
        else:
            v.append(0)
    return(v)

#potential well
def potential_four(d):
    v = []
    for i in range(0,Nx+2):
        if i >= int((Nx+2)*.6) and i < int((Nx+2)*.6)+Nx*d:
            v.append(-2.45*10**5)
        else:
            v.append(0)
    return v
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#initializes and solves question one
#V is potential and k is wave momentum
def question_one(V,k,sigma2,name):
    print(V)
    k_0 = k
    C = 1.0/(np.pi*sigma2)**0.25
    #initialize psi arrays: timesteps are the rows, x grid points are the columns
    Re = [[0.0 for j in range(0,Nx+2)] for i in range(0,tot_steps+1)]
    Im = [[0.0 for j in range(0,Nx+2)] for i in range(0,tot_steps+1)]
    psi_sqr = [[0.0 for j in range(0,Nx+2)] for i in range(0,tot_steps+1)]
    psi_sqrx = [[0.0 for j in range(0,Nx+2)] for i in range(0,tot_steps+1)]
    psi_sqrx2 = [[0.0 for j in range(0,Nx+2)] for i in range(0,tot_steps+1)]

    norm = []
    normx = []
    normx2 = []
    #initialize the wavefunction
    for j in range(0,Nx+1):
        x = j*dx
        Re[0][j] = C*np.exp((-(x-x_0)**2)/(4.0*sigma2))*np.cos(k_0*(x-x_0))
        Im[0][j] = -C*np.exp((-(x-x_0)**2)/(4.0*sigma2))*-np.sin(k_0*(x-x_0))
        psi_sqr[0][j] = Re[0][j]*Re[0][j] + Im[0][j]*Im[0][j]
        psi_sqrx[0][j] = Re[0][j]*j/(Nx+2)*Re[0][j] + Im[0][j]*j/(Nx+2)*Im[0][j]
        psi_sqrx2[0][j] = Re[0][j]*((j/(Nx+2))**2)*Re[0][j] + Im[0][j]*((j/(Nx+2))**2)*Im[0][j]    
    # add the current normalization to the list   
    norm.append(np.trapz(psi_sqr[0], dx= dx)-.414)
    normx.append(np.trapz(psi_sqrx[0], dx= dx))
    normx2.append(np.trapz(psi_sqrx2[0], dx = dx))
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #leapfrog loop
    for i in range(1,tot_steps):
        Im[i][0] = 0.0 
        Im[i][Nx+1] = 0.0
        Re[i][0] = 0.0
        Re[i][Nx+1] = 0.0
        for j in range(1,Nx):
            Im[i][j] = Im[i-1][j] + C1*(Re[i-1][j+1] -2.0*Re[i-1][j] + Re[i-1][j-1]) - C2*V[j]*Re[i-1][j]
        for j in range(1,Nx): 
            Re[i][j] = Re[i-1][j] - C1*(Im[i][j+1] -2.0*Im[i][j] + Im[i][j-1]) + C2*V[j]*Im[i][j]
            psi_sqr[i][j] = Re[i][j]*Re[i][j] + Im[i][j]*Im[i][j]
            psi_sqrx[i][j] = Re[i][j]*j/(Nx+2)*Re[i][j] + Im[i][j]*j/(Nx+2)*Im[i][j]
            psi_sqrx2[i][j] = Re[i][j]*((j/(Nx+2))**2)*Re[i][j] + Im[i][j]*((j/(Nx+2))**2)*Im[i][j]
        norm.append(np.trapz(psi_sqr[i],dx= dx)-.414)
        normx.append(np.trapz(psi_sqrx[i], dx= dx))
        normx2.append(np.trapz(psi_sqrx2[i], dx = dx))
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Calculate sigma(t)
    sigma = []
    for i in list(range(0, len(normx))):
        print(normx2[i],normx[i]**2,normx2[i] - normx[i]**2)
        sigma.append(np.abs(normx2[i] - normx[i]**2)**(1/2))
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Plots Normalization
    f,ax = plt.subplots()
    f.tight_layout()
    ax.plot(norm)
    ax.set_xlabel('time',fontsize=16)
    ax.set_ylabel('Normalization',fontsize=16)
    ax.set_title('Normailzation versus Time',style='italic',fontsize=16)
    ax.legend(loc="upper right")
    f.show()
    f.savefig("Norm"+name+".png", bbox_inches='tight',dpi=600)
    #Plots <x> vs time
    f1,ax1 = plt.subplots()
    f1.tight_layout()
    ax1.plot(normx)
    ax1.set_xlabel('time',fontsize=16)
    ax1.set_ylabel('<x>',fontsize=16)
    ax1.set_title('<x> versus time',style='italic',fontsize=16)
    ax1.legend(loc="upper right")
    f1.show()
    f1.savefig("X"+name+".png", bbox_inches='tight',dpi=600)
    #Plots <x^2> vs time
    f2,ax2 = plt.subplots()
    f2.tight_layout()
    ax2.plot(normx2)
    ax2.set_xlabel('time',fontsize=16)
    ax2.set_ylabel('$<x^2>$',fontsize=16)
    ax2.set_title('$<x^2>$'+' versus time',style='italic',fontsize=16)
    ax2.legend(loc="upper right")
    f2.show()
    f2.savefig("X2"+name+".png", bbox_inches='tight',dpi=600)
    #Plots sigma(t) vs time
    f4,ax4 = plt.subplots()
    f4.tight_layout()
    ax4.plot(sigma)
    ax4.set_xlabel('time',fontsize=16)
    ax4.set_ylabel('$\sigma(t)$',fontsize=16)
    ax4.set_title('$\sigma(t)$'+' versus time',style='italic',fontsize=16)
    ax4.legend(loc="upper right")
    f4.show()
    f4.savefig("sigma"+name+".png", bbox_inches='tight',dpi=600)
    #Plots psi at different points in time
    f3,ax3 = plt.subplots()
    f3.tight_layout()
    ax3.plot(psi_sqr[0],label='t=0')
    ax3.plot(psi_sqr[5000],label='t='+str(round(dt*5000,6)))
    ax3.plot(psi_sqr[7000],label='t='+str(round(dt*7000,6)))
    ax3.plot(V,'k',label='$V_0$')
    ax3.set_xticks(np.arange(0,1000,step=100))
    ax3.set_xticklabels([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1])
    ax3.set_ylim(0,60)
    ax3.set_xlabel('x',fontsize=16)
    ax3.set_ylabel('$\psi^*\psi$',fontsize=16)
    ax3.set_title('Wave Packet Position versus Time',style='italic',fontsize=16)
    ax3.legend(loc="upper right")
    f3.show()
    f3.savefig("Position"+name+".png", bbox_inches='tight',dpi=600)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#initializes and solves question three
#V is potential and k is wave momentum
def question_Two(V,sigma2):
    print(V)
    transmission = []
    for k in np.arange(450,600,5):
        k_0 = k
        C = 1.0/(np.pi*sigma2)**0.25
        steps =12000
        #initialize psi arrays: timesteps are the rows, x grid points are the columns
        Re = [[0.0 for j in range(0,Nx+2)] for i in range(0,steps+1)]
        Im = [[0.0 for j in range(0,Nx+2)] for i in range(0,steps+1)]
        psi_sqr = [[0.0 for j in range(0,Nx+2)] for i in range(0,steps+1)]
    
        norm = []
        #initialize the wavefunction
        for j in range(0,Nx+1):
            x = j*dx
            Re[0][j] = C*np.exp((-(x-x_0)**2)/(4.0*sigma2))*np.cos(k_0*(x-x_0))
            Im[0][j] = -C*np.exp((-(x-x_0)**2)/(4.0*sigma2))*-np.sin(k_0*(x-x_0))
            psi_sqr[0][j] = Re[0][j]*Re[0][j] + Im[0][j]*Im[0][j]  
        # add the current normalization to the list   
        norm.append(np.trapz(psi_sqr[0], dx= dx)-.414)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #leapfrog loop
        for i in range(1,steps):
            Im[i][0] = 0.0 
            Im[i][Nx+1] = 0.0
            Re[i][0] = 0.0
            Re[i][Nx+1] = 0.0
            for j in range(1,Nx):
                Im[i][j] = Im[i-1][j] + C1*(Re[i-1][j+1] -2.0*Re[i-1][j] + Re[i-1][j-1]) - C2*V[j]*Re[i-1][j]
            for j in range(1,Nx): 
                Re[i][j] = Re[i-1][j] - C1*(Im[i][j+1] -2.0*Im[i][j] + Im[i][j-1]) + C2*V[j]*Im[i][j]
                psi_sqr[i][j] = Re[i][j]*Re[i][j] + Im[i][j]*Im[i][j]
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        total=0
        for i in range(0,len(psi_sqr[-1])):
            if i>int((Nx+2)*.6)+Nx/1000*6:
                total+=psi_sqr[10000][i]
        #Calcualte transmission
        print(k,total/sum(psi_sqr[10000]))
        transmission.append(total/(sum(psi_sqr[10000])))
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #Plots psi at different points in time at the k of highest resonance
        if k==535:
                f3,ax3 = plt.subplots()
                f3.tight_layout()
                ax3.plot(psi_sqr[0],label='t=0')
                ax3.plot(psi_sqr[6500],label='t='+str(round(dt*6500,6)))
                ax3.plot(psi_sqr[10500],label='t='+str(round(dt*10500,6)))
                ax3.plot(V,'k',label='$V_0$')
                ax3.set_xticks(np.arange(0,1000,step=100))
                ax3.set_xticklabels([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1])
                ax3.set_ylim(0,30)
                ax3.set_xlabel('x',fontsize=16)
                ax3.set_ylabel('$\psi^*\psi$',fontsize=16)
                ax3.set_title('Wave Packet Position versus Time',style='italic',fontsize=16)
                ax3.legend(loc="upper right")
                f3.show()
                f3.savefig("Position3.png", bbox_inches='tight',dpi=600)
    #Plots the numeric Transmission coefficient for the double barrier
    maxX = np.arange(450,600,5)[np.argmax(transmission)]
    maxT = transmission[np.argmax(transmission)]
    f,ax = plt.subplots()
    f.tight_layout()
    ax.plot(np.arange(450,600,5),transmission)
    ax.annotate('max resonance', xy=(maxX, maxT), xytext=(470, .7),
            arrowprops=dict(facecolor='black', shrink=0.05)
            )
    ax.set_xlabel('k_0',fontsize=16)
    ax.set_ylabel('Transmission Coefficient',fontsize=16)
    ax.set_title('Numeric Transmission Coefficient',style='italic',fontsize=16)
    ax.legend(loc="upper right")
    f.show()
    f.savefig("T_numeric.png", bbox_inches='tight',dpi=600)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#initializes and solves question four
#V is potential and k is wave momentum
def question_Three(sigma2,k):
    transmission = []
    t1 = []
    E=k**2/2
    v=-2.45*10**(5)
    for d in np.arange(.001,.011,.001):
        t1.append((4*E*(E+v))/(4*E*(E+v)+v**2*np.sin((2*(E+v))**(1/2)*d)**2))
        print(d)
        V = potential_four(d)
        k_0 = k
        C = 1.0/(np.pi*sigma2)**0.25
        steps =14000
        #initialize psi arrays: timesteps are the rows, x grid points are the columns
        Re = [[0.0 for j in range(0,Nx+2)] for i in range(0,steps+1)]
        Im = [[0.0 for j in range(0,Nx+2)] for i in range(0,steps+1)]
        psi_sqr = [[0.0 for j in range(0,Nx+2)] for i in range(0,steps+1)]
    
        norm = []
        #initialize the wavefunction
        for j in range(0,Nx+1):
            x = j*dx
            Re[0][j] = C*np.exp((-(x-x_0)**2)/(4.0*sigma2))*np.cos(k_0*(x-x_0))
            Im[0][j] = -C*np.exp((-(x-x_0)**2)/(4.0*sigma2))*-np.sin(k_0*(x-x_0))
            psi_sqr[0][j] = Re[0][j]*Re[0][j] + Im[0][j]*Im[0][j]  
        # add the current normalization to the list   
        norm.append(np.trapz(psi_sqr[0], dx= dx)-.414)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #leapfrog loop
        for i in range(1,steps):
            Im[i][0] = 0.0 
            Im[i][Nx+1] = 0.0
            Re[i][0] = 0.0
            Re[i][Nx+1] = 0.0
            for j in range(1,Nx):
                Im[i][j] = Im[i-1][j] + C1*(Re[i-1][j+1] -2.0*Re[i-1][j] + Re[i-1][j-1]) - C2*V[j]*Re[i-1][j]
            for j in range(1,Nx): 
                Re[i][j] = Re[i-1][j] - C1*(Im[i][j+1] -2.0*Im[i][j] + Im[i][j-1]) + C2*V[j]*Im[i][j]
                psi_sqr[i][j] = Re[i][j]*Re[i][j] + Im[i][j]*Im[i][j]
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        total=0
        for i in range(0,len(psi_sqr[-1])):
            if i>int((Nx+2)*.6)+Nx/1000*6:
                total+=psi_sqr[10000][i]
        #Calcualte transmission
        print(k,total/sum(psi_sqr[10000]))
        transmission.append(total/(sum(psi_sqr[10000])))
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #Plots psi at different points in time at the d of highest resonance
        if d==.004:
                f3,ax3 = plt.subplots()
                f3.tight_layout()
                ax3.plot(psi_sqr[0],label='t=0')
                ax3.plot(psi_sqr[8200],label='t='+str(round(dt*8200,6)))
                ax3.plot(psi_sqr[13000],label='t='+str(round(dt*13000,6)))
                ax3.plot(V,'k',label='$V_0$')
                ax3.set_xticks(np.arange(0,1000,step=100))
                ax3.set_xticklabels([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1])
                ax3.set_ylim(0,25)
                ax3.set_xlim(200,900)
                ax3.set_xlabel('x',fontsize=16)
                ax3.set_ylabel('$\psi^*\psi$',fontsize=16)
                ax3.set_title('Wave Packet Position versus Time',style='italic',fontsize=16)
                ax3.legend(loc="upper right")
                f3.show()
                f3.savefig("Position4.png", bbox_inches='tight',dpi=600)
    #Plots the numeric Transmission coefficient for the double barrier
    f,ax = plt.subplots()
    f.tight_layout()
    ax.plot(np.arange(.001,.011,.001),transmission,label='numeric')
    ax.plot(np.arange(.001,.011,.001),t1,label='analytic')
    ax.set_xlabel('Barrier Depth '+'$(d)$',fontsize=16)
    ax.set_ylabel('Transmission Coefficient',fontsize=16)
    ax.set_title('Transmission Coefficient',style='italic',fontsize=16)
    ax.legend(loc="upper right")
    f.show()
    f.savefig("T_na.png", bbox_inches='tight',dpi=600)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Calculates the transmission coefficient as a function of k_0 and d
def transmissionD(k):
    Td = []
    E=(1/2)*k**2
    for d in np.arange(.001,.01,.0005):
        a=2*(d**2)*V*(1-E/V)
        num = np.sinh(a**(1/2))**2
        denum = 4*E/V*(1-E/V)
        Td.append((1+num/denum)**(-1))
    #Plots the transmission of k as a function of depth
    f1,ax1 = plt.subplots()
    f1.tight_layout()
    ax1.plot(np.arange(.001,.01,.0005),Td)
    ax1.set_xlabel('Barrier Depth (d)',fontsize=16)
    ax1.set_ylabel('Transmission Coefficeint (T)',fontsize=16)
    ax1.set_title('Transmission Coefficient versus Barrier Depth',style='italic',fontsize=16)
    f1.show()
    f1.savefig("TransmissionD.png", bbox_inches='tight',dpi=600)
###############################################################################
# MAIN CODE
###############################################################################
V1 = potential_one()
V2 = potential_two(.001)
V3 = potential_three()
#Transmission Coefficient Dependency on d
#transmissionD(700)
#Wave Packet with no potential interference
#question_one(V1, 700, 2.5*10**(-4),'1')
#question_one(V1, 700, 9.0*10**(-4),'1b')
#Wave Packet with a single barrier
#question_one(V2, 700, 2.5*10**(-4),'2')
#question_one(V2, 700, 2.5*10**(-4),'2b')
#Wave Packet with a double barrier
#question_Two(V3,.004)
#Wave packet with a potential well
question_Three(2.5*10**(-4),350)
