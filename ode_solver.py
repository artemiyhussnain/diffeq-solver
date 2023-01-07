import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')

step = 0.01
max_time = 10
timeplot = False
yplot = False
method = 'rk4' # rk4 or euler

f1 = lambda t, y: t*y**2 #y
f2 = lambda t, y, x: -1*t*x #x - x**3 - 0.25*y + 0.2645*np.sin(t)
x0 = 1 #0.9
y0 = 1 #0.4

h = step
t = np.arange(0, max_time + h, h)

x = np.zeros(len(t))
x[0] = x0

y = np.zeros(len(t))
y[0] = y0

for j in range(0, len(t) - 1):
    if method=='rk4':
        k1 = h * (f1(t[j], y[j]))
        k2 = h * (f1((t[j]+h/2), (y[j]+k1/2)))
        k3 = h * (f1((t[j]+h/2), (y[j]+k2/2)))
        k4 = h * (f1((t[j]+h), (y[j]+k3)))
        k = (k1+2*k2+2*k3+k4)/6
        x[j+1] = x[j] + k
        k1 = h * (f2(t[j], y[j], x[j]))
        k2 = h * (f2((t[j]+h/2), (y[j]+k1/2), (x[j]+k1/2)))
        k3 = h * (f2((t[j]+h/2), (y[j]+k2/2), (x[j]+k2/2)))
        k4 = h * (f2((t[j]+h), (y[j]+k3), (x[j]+k3)))
        k = (k1+2*k2+2*k3+k4)/6
        y[j+1] = y[j] + k
    if method=='euler':
        x[j + 1] = x[j] + h*f1(t[j], y[j])
        y[j + 1] = y[j] + h*f2(t[j], y[j], x[j])

plt.figure(figsize = (12, 8))
if timeplot:
    plt.xlabel('t')
    plt.ylabel('f(t)')
    plt.plot(t, x, 'bo', label='x', markersize=3)
    if yplot:
        plt.plot(t, y, 'ro', label='y', markersize=3)
else:
    plt.plot(x, y, 'bo', label='f', markersize=3)
    plt.xlabel('x')
    plt.ylabel('y')

plt.grid()
plt.legend(loc='lower right')
plt.show()
