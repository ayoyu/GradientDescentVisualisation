"""
@author:AYOUB EL KHALLIOUI
Created on Tue March 24 10:15:23 2018
"""
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib.animation import FuncAnimation


def gradient_descent_stream(function):
        nb_itermax = 100
        eps = 0.0001
        learning_rate = 0.1
        x0 = 35
        y0 = function(x0)
        cond = eps+10.
        nb_iter = 0.
        y_tmp = y0

        while cond > eps and nb_iter < nb_itermax:
                print('x=%.3f => y=%.3f and cond=%.3f'%(x0, y_tmp, cond))
                yield x0, y_tmp
                x0 = x0 - learning_rate * misc.derivative(function, x0)
                y0 = function(x0)
                cond = np.abs(y_tmp - y0)
                y_tmp = y0
                nb_iter += 1

if __name__ == '__main__':
        function = lambda x: 3 * x**2 + 2 * x + 1
        stream = gradient_descent_stream(function)
        x_func = np.arange(-15., 15, 1)
        y_func = [function(a) for a in x_func]

        def animate(i):
                try:
                        x, y = next(stream)
                        plt.cla()
                        plt.plot(x_func, y_func, 'r-')
                        plt.scatter(x, y)
                        plt.grid()
                        plt.title('Gradient descent 1D')
                        plt.tight_layout()
                except StopIteration:
                        print("End iterations for the gradient descent 1D")
                        sys.exit(0)
        
        
        ani = FuncAnimation(plt.gcf(), animate, interval=100)
        #plt.show()
        ani.save('./figures/GradientDescent1D.gif', writer='imagemagick', fps=2)