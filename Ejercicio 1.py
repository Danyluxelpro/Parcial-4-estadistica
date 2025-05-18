import numpy as np
from numpy import random as rd
import math
from scipy.stats import logistic
from scipy.stats import chisquare
from matplotlib import pyplot as plt
mods=["Logi"]
theta=[0]
tama=[50,100,200,500,1000]
data={}
estim=["-2ln","Wn","Rn"]

def flog(a,xi):
    res=(np.exp(xi-a)/(1+np.exp(xi-a))**2)
    return res

def fun(a,dat):
    dlog=sum((np.exp(y-a)-1)/(1+np.exp(y-a)) for y in dat)
    d2log=-2*sum((np.exp(y-a))/((1+np.exp(y-a))**2) for y in dat)
    fin=a-dlog/d2log
    return fin


for th in theta:
    for i in tama:
        data["Log"+str(th)+ " "+ str(i)+" MLE"]=[]
        data["Log"+str(th)+ " "+ str(i)+" -2ln"]=[]
        data["Log"+str(th)+ " "+ str(i)+" Wn"]=[]
        data["Log"+str(th)+ " "+ str(i)+" Rn"]=[]
        j=0
        while j<100:
            x=logistic.rvs(size=i)
            ave=sum(x)/i
            fave=fun(ave,x)
            m=0
            while abs(ave-fave)>=10**(-8) and m<=100:
                ave=fave
                fave=fun(ave,x)
                m+=1
            Wn=float(i*(1/3)*fave**2)
            Rn=float((3/i)*(sum((np.exp(y)-1)/(1+np.exp(y)) for y in x))**2)
            lamb=float(-2*sum(np.log(flog(y,0)/flog(y,fave)) for y in x))

            data["Log"+str(th)+ " "+ str(i)+" MLE"].append(float(fave))
            data["Log"+str(th)+ " "+ str(i)+" -2ln"].append(lamb)
            data["Log"+str(th)+ " "+ str(i)+" Wn"].append(Wn)
            data["Log"+str(th)+ " "+ str(i)+" Rn"].append(Rn)
            print(j)
            j+=1
    


quant=[0.032404,0.134029,0.32029,0.62669,1.1397,2.1469]
for i in tama:
    for est in estim:
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
        K=sum((y-i/7)**2/(i/7) for y in Y)
        data["Log"+str(th)+ " "+ str(i)+" "+est+" K"]=K
        data["Log"+str(th)+ " "+ str(i)+" "+est+" val"]=Y
        print("Resultado chi para "+str(th)+ " "+ str(i)+" "+est)
        print(chisquare(Y,ddof=1))

        

     
