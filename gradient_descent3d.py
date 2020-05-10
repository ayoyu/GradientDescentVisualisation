"""
@author: Ayoub El khallioui
Created on Tue March 24 13:26:17 2018
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
import sys
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


def partial_derivative(func, var=0, point=[]):
        args = point[:]
        def wrapper(x):
                args[var] = x
                return func(*args)
        return misc.derivative(wrapper, point[var], dx=1e-6)


def gradient_descent_stream(function):
        nb_iter = 0
        nb_itermax = 100
        eps = 0.0001
        alpha = 0.1
        x0_1 = -2.0
        x0_2 = 4.5
        x0_3 = 6.0
        cond = eps+10
        z0 = function(x0_1,x0_2,x0_3)
        z_tmp = z0
        while cond > eps and nb_iter < nb_itermax:
                print('x1 = %.3f x2 = %.3f x3 = %.3f => z_tmp = %.3f with cond=%.3f' % (x0_1, x0_2, x0_3, z_tmp, cond))
                yield x0_1, x0_2, x0_3
                x0_1 = x0_1 - alpha * partial_derivative(function, 0, [x0_1, x0_2, x0_3])
                x0_2 = x0_2 - alpha * partial_derivative(function, 1, [x0_1, x0_2, x0_3])
                x0_3 = x0_3 - alpha * partial_derivative(function, 2, [x0_1, x0_2, x0_3])
                z0 = function(x0_1, x0_2, x0_3)
                cond = np.abs(z_tmp - z0)
                z_tmp = z0


if __name__ == '__main__':
        x_sphere1 = 1.5
        x_sphere2 = 2.5
        x_sphere3 = 0.5
        function = lambda x1, x2, x3: (x1 - x_sphere1)**2 + (x2 - x_sphere2)**2 + (x3 - x_sphere3)**2
        stream = gradient_descent_stream(function)
        xf1 = np.arange(0.0, 3., 0.1)
        xf2 = np.arange(0.0, 3., 0.1)
        xf3 = np.arange(0.0, 3., 0.1)

        xxf1, xxf2 = np.meshgrid(xf1, xf2)
        _, xxf3 = np.meshgrid(xf1, xf3)
        zf = function(xxf1, xxf2, xxf3)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        def animate(i):
                try:
                        x1, x2, x3 = next(stream)
                        plt.cla()
                        ax.contour3D(xf1, xf2, zf)
                        ax.scatter(x1, x2, x3, c='red')
                        plt.title('Gradient descent 3D')
                        plt.tight_layout()
                except StopIteration:
                        print("End iterations for the gradient descent 3D")
                        sys.exit(0)

        ani = FuncAnimation(plt.gcf(), animate, interval=100)
        #plt.show()
        ani.save('./figures/GradientDescent3D.gif', writer='imagemagick', fps=5)