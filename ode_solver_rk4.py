import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')

def twovar(step=0.1, max_time=60, timeplot=True, plot_y = False):
    h = step
    t = np.arange(0, max_time + h, h)

    f1 = lambda t, y: y
    f2 = lambda t, x, y: 0.25*y + 0.2645*np.sin(t) - x
    x0 = 0.9
    y0 = 0.4

    x = np.zeros(len(t))
    x[0] = x0
    y = np.zeros(len(t))
    y[0] = y0

    for j in range(0, len(t) - 1):
        k1 = h * (f1(t[j], x[j]))
        k2 = h * (f1((t[j]+h/2), (x[j]+k1/2)))
        k3 = h * (f1((t[j]+h/2), (x[j]+k2/2)))
        k4 = h * (f1((t[j]+h), (x[j]+k3)))
        k = (k1+2*k2+2*k3+k4)/6
        x[j+1] = x[j] + k
        
        k1 = h * (f2(t[j], x[j], y[j]))
        k2 = h * (f2((t[j]+h/2), (x[j]+k1/2), (y[j]+k1/2)))
        k3 = h * (f2((t[j]+h/2), (x[j]+k2/2), (y[j]+k2/2)))
        k4 = h * (f2((t[j]+h), (x[j]+k3), (y[j]+k3)))
        k = (k1+2*k2+2*k3+k4)/6
        y[j+1] = y[j] + k

    plt.figure(figsize = (12, 8))
    if timeplot:
        plt.plot(t, x, 'bo--', label='x', markersize=6)
        if plot_y:
            plt.plot(t, y, 'ro--', label='y', markersize=6)
    else:
        plt.plot(x, y, 'bo--', label='f')

    plt.xlabel('t')
    plt.ylabel('f(t)')
    plt.grid()
    plt.legend(loc='lower right')
    plt.show()

twovar()

##            k1 = h * (f(t[j], x[j]))
##            k2 = h * (f((t[j]+h/2), (x[j]+k1/2)))
##            k3 = h * (f((t[j]+h/2), (x[j]+k2/2)))
##            k4 = h * (f((t[j]+h), (x[j]+k3)))
##            k = (k1+2*k2+2*k3+k4)/6
##            x[j+1] = x[j] + k
