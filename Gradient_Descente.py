"""
@author:AYOUB EL KHALLIOUI
Created on Tue March 24 10:15:23 2018
"""

from scipy import misc
import matplotlib.pyplot as plt
import numpy as np

def function(x):
        return 3*x**2+2*x+1

x=np.arange(-5.,5,1)
y=[]
for a in x:
    y.append(function(a))
plt.plot(x,y,'r-')


nb_itermax=100
eps=0.0001#condition d'arrét
learning_rate=0.1
x0=1.5
y0=function(x0)
plt.scatter(x0,function(x0))

cond=eps+10.
nb_iter=0.
y_tmp=y0

while cond>eps and nb_iter<nb_itermax:
        x0=x0-learning_rate*misc.derivative(function,x0)
        y0=function(x0)
        nb_iter=nb_iter+1
        cond=np.abs(y_tmp-y0)
        y_tmp=y0
        print('x0=%.3f=>y_tmp=%.3f and cond=%.3f'%(x0,y_tmp,cond))
        plt.scatter(x0,y_tmp)
plt.title('Gradient descent 1D')
plt.grid()
plt.show()








