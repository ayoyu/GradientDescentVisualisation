"""
@author: Ayoub El khallioui
Created on Tue March 24 11:42:16 2018
"""
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib.animation import FuncAnimation


def partial_derivative(func, var=0, point=[]):
        args=point[:]
        def wrapper(x):
                args[var] = x
                return func(*args)
        return misc.derivative(wrapper, point[var], dx=1e-6)


def gradient_descent_stream(function):
        nb_itermax = 100
        nb_iter = 0
        eps = 0.0001
        cond = 10 + eps
        x0_1 = 1.
        x0_2 = 1.5
        z0 = function(x0_1, x0_2)
        alpha = 0.1
        z_tmp = z0

        while cond > eps and nb_iter < nb_itermax:
                print('x1 = %.3f and x2 = %.3f => z = %.3f with cond = %.3f' % (x0_1, x0_2, z_tmp, cond))
                yield x0_1, x0_2
                x0_1 = x0_1 - alpha * partial_derivative(function , 0, [x0_1, x0_2])
                x0_2 = x0_2 - alpha * partial_derivative(function, 1, [x0_1, x0_2])
                z0 = function(x0_1, x0_2)
                cond = np.abs(z0 - z_tmp)
                z_tmp = z0
                nb_iter += 1


if __name__ == '__main__':
        function = lambda x1, x2: -np.exp(-x1**2 - x2**2)
        stream = gradient_descent_stream(function)
        xf1 = np.arange(-2., 2., 0.1)
        xf2 = np.arange(-2., 2., 0.1)
        xx1, xx2 = np.meshgrid(xf1, xf2)
        zf = function(xx1, xx2)
        
        def animate(i):
                try:
                        x1, x2 = next(stream)
                        plt.cla()
                        plt.contourf(xf1, xf2, zf)
                        plt.scatter(x1, x2, c='red')
                        plt.grid()
                        plt.title('Gradient descent 2D')
                        plt.tight_layout()
                except StopIteration:
                        print("End iterations for the gradient descent 2D")
                        sys.exit(0)
        
        
        ani = FuncAnimation(plt.gcf(), animate, interval=100)
        #plt.show()
        ani.save('./figures/GradientDescent2D.gif', writer='imagemagick', fps=5)