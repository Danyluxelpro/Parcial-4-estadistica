import numpy as np
from numpy import random as rd
import math
from scipy.stats import logistic,chi2
from scipy.stats import chisquare
from matplotlib import pyplot as plt
from scipy.stats import kstest
import statistics
#INFORMACIÓN

mods=["Logi"]
theta=[0]
tama=[50,100,200,500,1000]
data={}
estim=["-2ln","Wn","Rn"]
tam=1000
datoschi=chi2.rvs(df=1,size=tam)
#Se definen las densidades y la recursión de NR
def flog(a,xi):
    res=(np.exp(xi-a)/(1+np.exp(xi-a))**2)
    return res
def fun(a,dat):

    dlog=sum((np.exp(y-a)-1)/(1+np.exp(y-a)) for y in dat)
    d2log=-2*sum((np.exp(y-a))/((1+np.exp(y-a))**2) for y in dat)
    fin=a-dlog/d2log
    return fin

#Se generan los datos y se calculan lo estimadores
for th in theta:
    for i in tama:
        data["Log"+str(th)+ " "+ str(i)+" MLE"]=[]
        data["Log"+str(th)+ " "+ str(i)+" -2ln"]=[]
        data["Log"+str(th)+ " "+ str(i)+" Wn"]=[]
        data["Log"+str(th)+ " "+ str(i)+" Rn"]=[]
        j=0
        NR=[]
        
        #Newton Raphson con valor inicial el promedio
        while j<tam:
            x=logistic.rvs(size=i)
            ave=sum(x)/i
            fave=fun(ave,x)
            m=0
            while abs(ave-fave)>=10**(-5) and m<=100:
                ave=fave
                fave=fun(ave,x)
                m+=1
            NR.append(m)
            Wn=float(i*(1/3)*fave**2)
            Rn=float((3/i)*(sum((np.exp(y)-1)/(1+np.exp(y)) for y in x))**2)
            lamb=float(-2*sum(np.log(flog(y,0)/flog(y,fave)) for y in x))

            data["Log"+str(th)+ " "+ str(i)+" MLE"].append(float(fave))
            data["Log"+str(th)+ " "+ str(i)+" -2ln"].append(lamb)
            data["Log"+str(th)+ " "+ str(i)+" Wn"].append(Wn)
            data["Log"+str(th)+ " "+ str(i)+" Rn"].append(Rn)
            
            j+=1
    

    
#Pruebas de Pearson. Se hacen 7 celdas con las probabilidades de una chi2 de 6 grados de libertad 
#Pruebas de Pearson. Se hacen 7 celdas con las probabilidades de una chi2 de 6 grados de libertad 
print("Prueba Pearson 7 celdas")
quant=[0.032404,0.134029,0.32029,0.62669,1.1397,2.1469]
for est in estim:
    print("Estimador:"+est)
    for i in tama:
        print("Tamaño:"+str(i))
    
        Y=[0,0,0,0,0,0,0]
        for l in data["Log"+str(th)+ " "+ str(i)+" "+est]:
            if l<=quant[0]:
                Y[0]+=1
            elif quant[0]<l<=quant[1]:
                Y[1]+=1
            elif quant[1]<l<=quant[2]:
                Y[2]+=1
            elif quant[2]<l<=quant[3]:
                Y[3]+=1
            elif quant[3]<l<=quant[4]:
                Y[4]+=1
            elif quant[4]<l<=quant[5]:
                Y[5]+=1
            elif quant[5]<l:
                Y[6]+=1
        K=sum(((y-tam/7)**2)/(tam/7) for y in Y)
        data["Log"+str(th)+ " "+ str(i)+" "+est+" K"]=K
        data["Log"+str(th)+ " "+ str(i)+" "+est+" val"]=Y
        
        
        print("Resultado chi para "+str(th)+ " "+ str(i)+" "+est)
        print(Y)
        if K>=12.5918:
            print("Como K="+str(K)+" es mayor al valor de una chi cuadrado de 6 grados el cual es 12.5918 entonces se rechaza la hipotesis H0")
        else:
            print("Como K="+str(K)+" es menor al valor de una chi cuadrado de 6 grados el cual es 12.5918 entonces se acepta la hipotesis H0 ")
        print(chisquare(Y))
        
    print("---------------------------------------------------------")


for est in estim:
    
    for i in tama:
        print("Resultado K-S para Estimador:"+est+" Tamaño:"+str(i))
        
        pvalue=kstest(data["Log"+str(th)+ " "+ str(i)+" "+est],datoschi)[1]
        if pvalue>=0.05:
            print("Como pvalue="+str(pvalue)+" es mayor a 0.05  entonces se acepta la hipotesis H0")
        else:
            print("Como pvalue="+str(pvalue)+" es menor a 0.05 entonces se rechaza la hipotesis H0 ")
        print(kstest(data["Log"+str(th)+ " "+ str(i)+" "+est],datoschi))
        print()
    print("---------------------------------------------------------")


     
