#!/usr/bin/env python
# coding: utf-8

# In[13]:


#Empecemos por la colectividad ortogonal. El objetivo es simular varias matrices con numeros aleatorios, y obtener sus distintas
#distribuciones. Empezaremos siempre por simular una matriz, y luego pasaremos a simular varias matrices. Primero, observa todas
#las funciones que necesitaremos para esto, con las bibliotecas necesarias. 

from numpy import sqrt,log,cos,sin,pi,linspace
import math as mt
from random import random
import matplotlib.pyplot as plt
import numpy as np


def gaussian(mu,sigma):
    r=sqrt(-2*sigma**2*log(1-random()))
    theta=2*pi*random()
    x=mu + r*cos(theta)
    y=mu + r*sin(theta)
    
    return x,y
#Esto te genera un par de numeros gaussianos con la media y el valor que quieras. Para un codigo, estaría chachi que te cogiese sólo de un tipo
#Tenemos que ver como queremos construir cada elemento de la matriz. No todos tendrán las mismas características.
#Para construir una matriz NxN en Python, cada término con un número aleatorio
import numpy as np
def matriceador(N,mu,sigma):
    A=np.empty((N,N))
    for i in range(N):
        for j in range(N):
            A[i,j]=gaussian(mu,sigma)[0]
    return A
#Te saca una matriz de numeros aleatorios con un determinado sigma y un determinado mu. Puedes elegir las características de la
#distribución de los numeros
def matriceador2(N,mu,sigma):
    A=np.empty((N,N),dtype=complex)
    B=np.empty((N,N),dtype=complex)
    for i in range(N):
        for k in range(N):
            v=gaussian(mu,sigma)
            A[i,k]=complex(v[0])
            B[i,k]=complex(v[1])*1j
    return A, B
#Te saca dos matrices de numeros aleatorios con distribución normal. Ambas comparten mismas características de distribución
#aunque la forma de cambiar esto ultimo es relativamente sencilla. Util para el caso unitario
def relfreq(v,e):
    v=list(v);v=sorted(v); j=v.index(e)
    x=[]; i=0; #print(j)
    while i<=j:
        x.append(v.count(v[i]))
        i=i+v.count(v[i])
        #print(i)
    return x
#Te saca la degeneración de cada sistema. Es, realmente, la función de probabilidad relativa de la distribución de autovalores            


# In[4]:


#Veamos para una matriz en el caso ortogonal:
N=1000
A=matriceador(N,0,1)
A=1/2*(A+np.transpose(A))
autovals,autovects=np.linalg.eig(A)
a=autovals
e=list(a)
edagga=sorted(e)
x=np.linspace(edagga[1],edagga[len(edagga)-1],1000000)
y=(1/pi)*sqrt(2*N-x**2)
plt.hist(a,bins=90,color='green') #Unas 275 bins para una matriz de 10000, 90 bins para una de 1000
plt.plot(x,y,color='red')
plt.title('Densidad de Autovalores para 10000x10000. Ley del semicirculo')
plt.show()
#Hasta aquí hemos representado la ley del semicirculo de Wigner. Pero los resultados no son optimos. Trabajamos con la densidad acumulada de autovalores, que usaremos como variable del unfolding. 
from math import asin
adagga=sorted(a)
eind=adagga[len(adagga)-1]
eunfold=relfreq(a,eind)
eunfold=(1/N)*np.array(eunfold)
sigma=[]
for i in range(0,len(x)):
    sigma.append(asin(x[i]/sqrt(2*N))/pi+x[i]/(pi*np.sqrt(2*N))*np.sqrt(1-(x[i]/np.sqrt(2*N))**2)+1/2)
plt.plot(adagga,np.cumsum(eunfold),linewidth=0.8)
plt.plot(x,sigma,color='red',linewidth=0.8)
plt.hist(a,bins=100,color='green',density='True',cumulative='True')
plt.title('Densidad acumulada de autovalores. Variable del Unfolding')
plt.show()
eta=[]
for i in range(len(edagga)):
    eta.append(asin(adagga[i]/sqrt(2*N))/pi+adagga[i]/(pi*np.sqrt(2*N))*np.sqrt(1-(adagga[i]/np.sqrt(2*N))**2)+1/2)
    #print(i)
#print(eta)
#plt.plot(eta,np.cumsum(eunfold))
eta=np.array(eta)
s=[]
for i in range(0,(len(eta)-1)):
    s.append((N-1)*(eta[i+1]-eta[i])/(eta[N-1]-eta[0]))
#print(s)
#Ya tenemos la diferencia entre los autovalores. Pero ahora hay que calcular la probabilidad de que sea un valor, que será este
#valor, es decir, el numero de diferencia de autovalores compatibles entre el valor total, para que todo sume 1.
snorm=list(np.array(s)/len(s))
sordered=sorted(s)
sx=np.linspace(sordered[0],sordered[len(s)-1],10000)
ps=(pi/2)*sx*np.exp(-pi/4*sx**2)
plt.hist(s,bins=70,color='red',density='True')
plt.plot(sx,ps,color='black')
plt.title('Ley de Wigner de autovalores contiguos')
plt.show()


# In[5]:


#Y para una matriz en el caso unitario :)
N=1000
Ap,B=matriceador2(N,0,1/np.sqrt(2))
A=Ap+B
A=1/2*(A+np.transpose(np.conjugate(A)))
autovals,autovects=np.linalg.eig(A)
a=autovals
e=list(a)
#for i in range(0,len(e)-1):
    #e[i]=float(e[i])
edagga=sorted(e)
#print(edagga)
#print(edagga)
#print(len(edagga))
#Antes de plotear nada, tenemos que encontrar el numero de bins. Ploteemos el numero de autovalores, que hay 1000.
#Tenemos que ver que intervalo nos conviene elegir. Los autovalores van desde -44 a 44 aproximadamente. #Para la ley de wigner 
x=np.linspace(edagga[1],edagga[len(edagga)-1],1000000)
y=(1/pi)*sqrt(2*N-x**2)
plt.hist(edagga,bins=90,color='green') #Unas 275 bins para una matriz de 10000, 90 bins para una de 1000
plt.plot(x,y,color='red')
plt.title('Densidad de Autovalores para 10000x10000. Ley del semicirculo')
plt.show()
from math import asin
adagga=np.array(edagga)
eind=adagga[len(adagga)-1]
eunfold=relfreq(a,eind)
eunfold=(1/N)*np.array(eunfold)
sigma=[]
for i in range(0,len(x)):
    sigma.append(asin(x[i]/sqrt(2*N))/pi+x[i]/(pi*np.sqrt(2*N))*np.sqrt(1-(x[i]/np.sqrt(2*N))**2)+1/2)
plt.plot(adagga,np.cumsum(eunfold))
plt.plot(x,sigma,color='red')
plt.hist(a,bins=90,color='green',density='True',cumulative='True') #Unas 90 bins para 1000
plt.title('Densidad acumulada de autovalores. Variable del Unfolding')
plt.show()
eta=[]
for i in range(len(edagga)):
    eta.append(asin(adagga[i]/sqrt(2*N))/pi+adagga[i]/(pi*np.sqrt(2*N))*np.sqrt(1-(adagga[i]/np.sqrt(2*N))**2)+1/2)
    #print(i)
#print(eta)
#plt.plot(eta,np.cumsum(eunfold))
eta=np.array(eta)
s=[]
for i in range(0,(len(eta)-1)):
    s.append((N-1)*(eta[i+1]-eta[i])/(eta[N-1]-eta[0]))
#print(s)
#Ya tenemos la diferencia entre los autovalores. Pero ahora hay que calcular la probabilidad de que sea un valor, que será este
#valor, es decir, el numero de diferencia de autovalores compatibles entre el valor total, para que todo sume 1.
snorm=list(np.array(s)/len(s))
sordered=sorted(s)
sx=np.linspace(sordered[0],sordered[len(s)-1],10000)
ps=(32/pi**2)*(sx**2)*np.exp(-4/pi*sx**2)
plt.hist(s,bins=70,color='red',density='True')
plt.plot(sx,ps,color='black')
plt.title('Ley de Wigner de autovalores contiguos')
plt.show()


# In[31]:


#Ahora toca inventarse algún método para hacer este análisis en un conjunto N de matrices. Creo que voy a crear una función 
#tochísima que me haga este análisis, tanto para las s como para las densidades. Simplemente le meta el tipo de matriz y me haga
#todo el analisis directamente. Eso es porque en el fondo, no necesitas saber que ocurre "dentro", son siempre numeros aleatorios
#Vamos a crear una función que te genere listas de matrices para poder diagonalizarlas, y hacer todo el curro con ellas.
def PabloORTO(mu,sigma,N,L):  #N es la dimensión de la matriz, y L la longitud que quieras del vector
    M=[]
    i=0
    while i<L:
        A=matriceador(N,mu,sigma)
        M.append(1/2*(A+np.transpose(A)))
        i=i+1
    if L!=len(M):
        print("Algo está fallando en el codigo :(:(:(")
    else:
        list(M)
        k=0
        Energias=[]
        EnergiasUnfolded=[]
        while k<L:
            ener,a=np.linalg.eig(M[k])
            Energias=Energias+list(ener)
            eta=[]
            adagga=sorted(ener)
            o=0
            while o<len(ener):
                if 1<abs((adagga[o]/np.sqrt(2*N))):
                    print("El autovalor de indice",o,"es mayor que 1 (En la matriz",k,"del vector matrices)")
                    o=o+1
                else:
                    eta.append(mt.asin(adagga[o]/np.sqrt(2*N))/pi+adagga[o]/(pi*np.sqrt(2*N))*np.sqrt(1-(adagga[o]/np.sqrt(2*N))**2)+1/2)
                    o=o+1
            eta=np.array(eta)
            s=[]
            for z in range(0,(len(eta)-1)):
                s.append((N-1)*(eta[z+1]-eta[z])/(eta[len(eta)-1]-eta[0]))
            EnergiasUnfolded=EnergiasUnfolded+list(s)
            k=k+1
    Energias=sorted(list(Energias))
    EnergiasUnfolded=sorted(list(EnergiasUnfolded))
    return Energias, EnergiasUnfolded

#Para el caso unitario, sólo tendremos que cambiar a priori la forma de generar matrices. Queda como ejercicio futuro meterle
#un anexo al codigo que te analice si la parte compleja de los autovalores pueden ser consideradas errores numericos.

def PabloUNIT(mu,sigma,N,L):  #N es la dimensión de la matriz, y L la longitud que quieras del vector
    M=[]
    i=0
    while i<L:
        Ap,B=matriceador2(N,0,1/np.sqrt(2))
        A=Ap+B
        M.append(1/2*(A+np.transpose(np.conjugate(A))))
        i=i+1
    if L!=len(M):
        print("Algo está fallando en el codigo :(:(:(")
    else:
        list(M)
        k=0
        Energias=[]
        EnergiasUnfolded=[]
        while k<L:
            ener,a=np.linalg.eig(M[k])
            Energias=Energias+list(ener)
            eta=[]
            adagga=sorted(ener)
            o=0
            while o<len(ener):
                if 1<abs((adagga[o]/np.sqrt(2*N))):
                    print("El autovalor de indice",o,"es mayor que 1 (En la matriz",k,"del vector matrices)")
                    o=o+1
                else:
                    eta.append(mt.asin(adagga[o]/np.sqrt(2*N))/pi+adagga[o]/(pi*np.sqrt(2*N))*np.sqrt(1-(adagga[o]/np.sqrt(2*N))**2)+1/2)
                    o=o+1
            eta=np.array(eta)
            s=[]
            for z in range(0,(len(eta)-1)):
                s.append((N-1)*(eta[z+1]-eta[z])/(eta[len(eta)-1]-eta[0]))
            EnergiasUnfolded=EnergiasUnfolded+list(s)
            k=k+1
    Energias=sorted(list(Energias))
    EnergiasUnfolded=sorted(list(EnergiasUnfolded))
    return Energias, EnergiasUnfolded


#Ya tenemos nuestro generador de matrices vectoriales. Esto nos permitirá simplemente generar L matrices de un ensemble y 
# hacer estadística con los autovalores de todas. 


# In[17]:


#Simulemos tanto las variables de la energía como las s del unfolding, en ambos ensembles. Así podremos ponernos a hacer plots
#calculitos guapos guapos. Una vez tengamos esto, sólo faltará definir las variables y hacer los plots. Ya estamos muy cerca de
#acabar. Observa que el tiempo de iteración aquí es menor que si hicieses una matriz 10000x10000, por lo que computacionalmente, 
#esto es más efectivo.
N=1000;  L=30
e,s=PabloORTO(0,1,N,L)
x=np.linspace(e[0],e[len(e)-1],10000)
y=y=(1/pi)*sqrt(2*N-x**2)
plt.hist(e,bins=3450,color='green') #Unas 275 bins para una matriz de 10000, 90 bins para una de 1000
plt.plot(x,y,color='red')
plt.title('Densidad de Autovalores. Ley del semicirculo')
plt.show()

eunfold=relfreq(e,e[len(e)-1])
eunfold=(1/(N*L))*np.array(eunfold)

sigma=[]
i=0
b=[]
while i<len(x):
    if 1<abs((x[i]/np.sqrt(2*N))):
        #print("El autovalor de indice",i,"es mayor que 1 (En la matriz",k,"del vector matrices)")
        i=i+1
    else:
        sigma.append(mt.asin(x[i]/np.sqrt(2*N))/pi+x[i]/(pi*np.sqrt(2*N))*np.sqrt(1-(x[i]/np.sqrt(2*N))**2)+1/2)
        b.append(x[i])
        i=i+1
plt.plot(e,np.cumsum(eunfold))
plt.plot(b,sigma,color='red')
plt.hist(e,bins=1000,color='green',density='True',cumulative='True') #Unas 90 bins para 1000
plt.title('Densidad acumulada de autovalores. Variable del Unfolding')
plt.show()

sx=np.linspace(s[0],s[len(s)-1],10000)
ps=(pi/2)*sx*np.exp(-pi/4*sx**2)
plt.hist(s,bins=925,color='red',density='True')
plt.plot(sx,ps,color='black')
plt.title('Ley de Wigner de autovalores contiguos')
plt.show()


# In[37]:


#Veamos el analisis para el caso unitario, ahora que tenemos el caso ortogonal

N=1000;  L=30
e2,s2=PabloUNIT(0,1/sqrt(2),N,L)
x=np.linspace(e[0],e[len(e)-1],10000)
y=(1/pi)*sqrt(2*N-x**2)
plt.hist(e,bins=3450,color='green') #Unas 275 bins para una matriz de 10000, 90 bins para una de 1000
plt.plot(x,y,color='red')
plt.title('Densidad de Autovalores. Ley del semicirculo')
plt.show()

eunfold2=relfreq(e2,e2[len(e2)-1])
eunfold2=(1/(N*L))*np.array(eunfold2)

sigma=[]
i=0
b=[]
while i<len(x):
    if 1<abs((x[i]/np.sqrt(2*N))):
        #print("El autovalor de indice",i,"es mayor que 1 (En la matriz",k,"del vector matrices)")
        i=i+1
    else:
        sigma.append(mt.asin(x[i]/np.sqrt(2*N))/pi+x[i]/(pi*np.sqrt(2*N))*np.sqrt(1-(x[i]/np.sqrt(2*N))**2)+1/2)
        b.append(x[i])
        i=i+1
plt.plot(e2,np.cumsum(eunfold2))
plt.plot(b,sigma,color='red')
plt.hist(e2,bins=1000,color='green',density='True',cumulative='True') #Unas 90 bins para 1000
plt.title('Densidad acumulada de autovalores. Variable del Unfolding')
plt.show()

sx=np.linspace(s2[0],s2[len(s2)-1],10000)
ps=(32/pi**2)*(sx**2)*np.exp(-4/pi*sx**2)
plt.hist(s2,bins=925,color='red',density='True')
plt.plot(sx,ps,color='black')
plt.title('Ley de Wigner de autovalores contiguos')
plt.show()



