import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-poster')

s_f = lambda t, s, i, r: 3 + 0.01*r - 0.0005*s*i
s0 = 990
i_f = lambda t, s, i: 0.0005*s*i - 0.05*i - 0.02*i
i0 = 10
r_f = lambda t, i, r: 0.05*i - 0.01*r
r0 = 0

h = 0.01
t = np.arange(0, 100 + h, h)

s = np.zeros(len(t))
s[0] = s0
i = np.zeros(len(t))
i[0] = s0
r = np.zeros(len(t))
r[0] = s0

for j in range(0, len(t) - 1):
    s[j + 1] = s[j] + h*s_f(t[j], s[j], i[j], r[j])
    i[j + 1] = i[j] + h*i_f(t[j], s[j], i[j])
    r[j + 1] = r[j] + h*r_f(t[j], i[j], r[j])

plt.figure(figsize = (12, 8))
plt.plot(t, s, 'bo--', label='s')
plt.plot(t, i, 'ro--', label='i')
plt.plot(t, r, 'go--', label='r')

plt.title('Approximate Solutions \
for Simple ODEs')
plt.xlabel('t')
plt.ylabel('f(t)')
plt.grid()
plt.legend(loc='lower right')
plt.show()
