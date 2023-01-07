import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-poster')
step = 0.1
max_time = 100
timeplot = True

f1 = lambda t, y: y
f2 = lambda t, x, y: x - x**3 - 0.25*y + 0.2645*np.sin(t)
x0 = 0.4
y0 = 0.9

h = step
t = np.arange(0, max_time + h, h)
x = np.zeros(len(t))
x[0] = x0
y = np.zeros(len(t))
y[0] = y0

for j in range(0, len(t) - 1):
    x[j + 1] = x[j] + h*f1(t[j], y[j])
    y[j + 1] = y[j] + h*f2(t[j], x[j], y[j])

plt.figure(figsize = (12, 8))
if timeplot:
    plt.plot(t, x, 'bo', label='x', markersize=6)
    #plt.plot(t, y, 'ro--', label='y', markersize=6)
else:
    plt.plot(x, y, 'bo--', label='f')

plt.xlabel('t')
plt.ylabel('f(t)')
plt.grid()
plt.legend(loc='lower right')
plt.show()
