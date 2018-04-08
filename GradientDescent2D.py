"""
@author: Ayoub El khallioui
Created on Tue March 24 11:42:16 2018
"""

from scipy import misc
import matplotlib.pyplot as plt
import numpy as np

def function_2D(x1,x2):
        return -1.*np.exp(-x1**2-x2**2)
x1=np.arange(-2.,2.,0.1)
x2=np.arange(-2.,2.,0.1)
xx1,xx2=np.meshgrid(x1,x2)
z=function_2D(xx1,xx2)
plt.contourf(x1,x2,z)


def partial_derivative(func,var=0,point=[]):
        args=point[:]
        def war(x):
                args[var]=x
                return func(*args)
        return misc.derivative(war,point[var],dx=1e-6)

nb_itermax=100
nb_iter=0.
eps=0.0001
cond=10+eps
x0_1=1.
x0_2=1.5
z0=function_2D(x0_1,x0_2)
plt.scatter(x0_1,x0_2)
alpha=0.1 #learning_rate
z_tmp=z0

while cond>eps and nb_iter<nb_itermax:
        x0_1=x0_1-alpha*partial_derivative(function_2D,0,[x0_1,x0_2])
        x0_2=x0_2-alpha*partial_derivative(function_2D,1,[x0_1,x0_2])
        z0=function_2D(x0_1,x0_2)
        cond=np.abs(z0-z_tmp)
        z_tmp=z0
        nb_iter=nb_iter+1
        plt.scatter(x0_1,x0_2)
        print('x0_1=%.3f and x0_2=%.3f =>z_tmp=%.3f with cond=%.3f'%(x0_1,x0_2,z_tmp,cond))
plt.title('Gradient descent 2D')
plt.show()











