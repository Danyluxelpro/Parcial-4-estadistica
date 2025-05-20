import numpy as np
from numpy import random as rd
import math

from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import laplace, t
from scipy.stats import wilcoxon
from scipy.stats import ttest_ind
np.random.seed(123)
n = 60
m = 60         
B = 1000              
alpha = 0.05         
     #laplace   bootstrap
#Bajo Delta =0
print("Laplace delta 0")
X = laplace.rvs(loc=0, scale=1, size=n)
Y = laplace.rvs(loc=0, scale=1, size=m)  
T_obs = np.mean(Y) - np.mean(X)
Z = np.concatenate([X, Y])
T_boot = np.zeros(B)

for i in range(B):
    X_star = np.random.choice(Z, size=n, replace=True)
    Y_star = np.random.choice(Z, size=m, replace=True)
    T_boot[i] = np.mean(Y_star) - np.mean(X_star)

p_value = np.mean(T_boot >= T_obs)
print("p-valor (Laplace) bootstrap: "+str(p_value))

if p_value < alpha:
    print("Decisión: Rechazar H0")
else:
    print("Decisión: No rechazar H0")
print(" ")
#Laplace wilcoxon
p_wilc=wilcoxon(X,Y).pvalue
print("p-valor (Laplace) wilcoxon: "+str(p_wilc))
if p_wilc < alpha:
    print("Decisión: Rechazar H0")
else:
    print("Decisión: No rechazar H0")
print(" ")
#Laplace ttest
p_tt=ttest_ind(X,Y).pvalue
print("p-valor (Laplace) ttest: "+str(p_tt))
if p_tt < alpha:
    print("Decisión: Rechazar H0")
else:
    print("Decisión: No rechazar H0")
print(" ")
#Bajo delta=0.3
print("Laplace delta 0.3")
X = laplace.rvs(loc=0, scale=1, size=n)
Y = laplace.rvs(loc=0.3, scale=1, size=m) 
V_obs = np.mean(Y) - np.mean(X)
Z = np.concatenate([X, Y])
V_boot = np.zeros(B)

for i in range(B):
    X_star = np.random.choice(Z, size=n, replace=True)
    Y_star = np.random.choice(Z, size=m, replace=True)
    V_boot[i] = np.mean(Y_star) - np.mean(X_star)

p_value = np.mean(V_boot >= V_obs)
print("p-valor (Laplace) bootstrap: "+str(p_value))

if p_value < alpha:
    print("Decisión: Rechazar H0")
else:
    print("Decisión: No rechazar H0")
print(" ")

p_wilc=wilcoxon(X,Y).pvalue
print("p-valor (Laplace) wilcoxon: "+str(p_wilc))
if p_wilc < alpha:
    print("Decisión: Rechazar H0")
else:
    print("Decisión: No rechazar H0")
print(" ")
p_tt=ttest_ind(X,Y).pvalue
print("p-valor (Laplace) ttest: "+str(p_tt))
if p_tt < alpha:
    print("Decisión: Rechazar H0")
else:
    print("Decisión: No rechazar H0")
print(" ")
#T-student
#bajo delta=0

for r in [2, 10, 50]:
    print("Grados de libertad (r): "+str(r)+" delta 0")
    

  
    X = t.rvs(df=r, size=n)
    Y = t.rvs(df=r, size=m)   

  
    V_obs = np.mean(Y) - np.mean(X)

    
    Z = np.concatenate([X, Y])
    V_boot = np.zeros(B)

    for i in range(B):
        X_star = np.random.choice(Z, size=n, replace=True)
        Y_star = np.random.choice(Z, size=m, replace=True)
        V_boot[i] = np.mean(Y_star) - np.mean(X_star)


    p_value = np.mean(V_boot >= V_obs)
    print("p-valor (Tstudent) boostrap: "+str(p_value))

    if p_value < alpha:
        print("Decisión: Rechazar H0")
    else:
        print("Decisión: No rechazar H0")
    print(" ")
    p_wilc=wilcoxon(X,Y).pvalue
    print("p-valor (Tstudent) wilcoxon: "+str(p_wilc))
    if p_wilc < alpha:
        print("Decisión: Rechazar H0")
    else:
        print("Decisión: No rechazar H0")
    print(" ")
    p_tt=ttest_ind(X,Y).pvalue
    print("p-valor (Tstudent) ttest: "+str(p_tt))
    if p_tt < alpha:
        print("Decisión: Rechazar H0")
    else:
        print("Decisión: No rechazar H0")
    print(" ")
#bajo delta=0.3
for r in [2, 10, 50]:
    print("Grados de libertad (r): "+str(r)+ " delta 0.3")

  
    X = t.rvs(df=r, size=n)
    Y = t.rvs(df=r, size=m) + 0.3 

  
    V_obs = np.mean(Y) - np.mean(X)

    
    Z = np.concatenate([X, Y])
    V_boot = np.zeros(B)

    for i in range(B):
        X_sim = np.random.choice(Z, size=n, replace=True)
        Y_sim = np.random.choice(Z, size=m, replace=True)
        V_boot[i] = np.mean(Y_sim) - np.mean(X_sim)


    p_value = np.mean(V_boot >= V_obs)
    print("p-valor (Tstudent) boostrap: "+str(p_value))

    if p_value < alpha:
        print("Decisión: Rechazar H0")
    else:
        print("Decisión: No rechazar H0")
    print(" ")
    p_wilc=wilcoxon(X,Y).pvalue
    print("p-valor (Tstudent) wilcoxon: "+str(p_wilc))
    if p_wilc < alpha:
        print("Decisión: Rechazar H0")
    else:
        print("Decisión: No rechazar H0")
    print(" ")
    
    p_tt=ttest_ind(X,Y).pvalue
    print("p-valor (Tstudent) ttest: "+str(p_tt))
    if p_tt < alpha:
        print("Decisión: Rechazar H0")
    else:
        print("Decisión: No rechazar H0")
    print(" ")




