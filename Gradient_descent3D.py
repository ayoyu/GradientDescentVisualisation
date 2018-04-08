"""
@author: Ayoub El khallioui
Created on Tue March 24 13:26:17 2018
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import misc

global x_sphere1
global x_sphere2
global x_sphere3
x_sphere1=1.5
x_sphere2=2.5
x_sphere3=0.5

def function_3D(x1,x2,x3):
        r1=x1-x_sphere1
        r2=x2-x_sphere2
        r3=x3-x_sphere3
        return r1**2+r2**2+r3**2

def partial_derivative(func,var=0,point=[]):
        args=point[:]
        def war(x):
                args[var]=x
                return func(*args)
        return misc.derivative(war,point[var],dx=1e-6)
###plot function 
x1=np.arange(0.0,3.,0.1)
x2=np.arange(0.0,3.,0.1)
x3=np.arange(0.0,3.,0.1)

dim_x1=x1.shape[0]
dim_x2=x2.shape[0]
dim_x3=x3.shape[0]

z12=np.zeros((dim_x1,dim_x2))
z13=np.zeros((dim_x2,dim_x3))

for i in np.arange(dim_x1):
        for j in np.arange(dim_x2):
                r1=x1[i]-x_sphere1
                r2=x2[j]-x_sphere2
                r3=0.0-x_sphere3
                z12[i,j]=r1**2+r2**2+r3**2
plt.contourf(x1,x2,z12)
#plt.show()
for i in np.arange(dim_x1):
        for j in np.arange(dim_x3):
                r1=x1[i]-x_sphere1
                r2=0.0-x_sphere2
                r3=x3[j]-x_sphere3
                z13[i,j]=r1**2+r2**2+r3**2
plt.contourf(x1,x2,z13)
#plt.show()

nb_iter=0
nb_itermax=100
eps=0.0001
alpha=0.1
x0_1=0.0
x0_2=0.5
x0_3=0.0
plt.scatter(x0_1,x0_2,x0_3)

cond=eps+10
z0=function_3D(x0_1,x0_2,x0_3)
z_tmp=z0
while cond>eps and nb_iter<nb_itermax:
        x0_1=x0_1-alpha*partial_derivative(function_3D,0,[x0_1,x0_2,x0_3])
        x0_2=x0_2-alpha*partial_derivative(function_3D,1,[x0_1,x0_2,x0_3])
        x0_3=x0_3-alpha*partial_derivative(function_3D,2,[x0_1,x0_2,x0_3])
        z0=function_3D(x0_1,x0_2,x0_3)
        cond=np.abs(z_tmp-z0)
        z_tmp=z0
        print('x0_1=%.3f x0_2=%.3f x0_3=%.3f with z_tmp=%.3f and cond=%.3f'%(x0_1,x0_2,x0_3,z_tmp,cond))
        plt.scatter(x0_1,x0_2,x0_3)
plt.title('Gradient descent 3D')
plt.show()



