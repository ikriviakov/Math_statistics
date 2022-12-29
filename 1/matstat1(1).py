import numpy as np
import math
import scipy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def print_matrix(x):
    for i in range(0, len(x)):
        for i2 in range(0, len(x[i])):
            print(x[i][i2], end=' ')
        print()

data =  [
    [14.495,4.715,7.175,8.428,11.093,3.375,12.906,8.415,8.916,13.4],
    [5.343,17.985,15.992,13.89,9.838,13.924,9.012,9.458,17.69,6.542],
    [14.396,8.592,8.206,14.237,7.357,10.821,12.767,16.058,12.959,4.354],
    [12.888,10.268,9.182,5.647,8.282,2.903,15.988,12.959,14.919,6.339],
    [2.375,17.921,9.097,15.85,11.449,11.095,9.493,12.175,7.479,13.535],
    [9.234,6.078,4.964,6.355,13.957,12.911,15.694,14.286,9.869,5.175],
    [5.811,7.241,5.814,3.086,6.875,3.878,5.333,15.134,12.924,9.159],
    [4.727,4.646,15.535,9.919,17.117,10.351,16.892,12.423,10.511,4.942],
    [4.843,9.927,15.864,3.635,17.963,8.25,5.14,6.734,12.622,13.325],
    [3.377,16.195,12.04,12.768,2.744,14.186,9.354,15.439,14.612,15.649],
    [8.681,5.006,3.608,2.867,12.177,15.506,7.683,14.022,17.103,8.905],
    [12.173,17.757,6.883,2.666,9.861,5.743,16.175,15.308,7.039,15.238]
    ]

data = np.array(data, float)
print("Выборка вариант 9")
print_matrix(data)

##

print("\n\nНаходим крайние члены вариационного ряда и размах выборки")
max = np.amax(data)
min = np.amin(data)
print("Крайние члены вариационного ряда:\nmax = ", max, "min = ", min)
w = max - min
print("Размах выборки:\nw = ", w)

n = np.size(data)
l = math.trunc(1 + math.log2(n))
# l = 15
h = w / l
print("Размер n = ",n,"\nчисло интервалов l=",l,"ширина интервалов h = ",h)
histogram,binEdges  = np.histogram(data,l)
binEdgesAverage = np.zeros(binEdges[:l].size)
for i in range (l):
    binEdgesAverage[i] = (binEdges[i] + binEdges[i + 1]) / 2 
print("Средние значения интервалов binEdgesAverage =",binEdgesAverage)
print("Итнтервалы", binEdges)
p = histogram / n
print("Относительные частоты p = ",np.around(p,5))
print("Частоты histogram = ", histogram)
f1 = p / h
print("f1 = ",np.around(f1,7))

##

sns.set_theme()
plt.figure(figsize=(9,6))
x = binEdgesAverage
y = f1
plt.bar(x, y, width=h)
plt.plot(x, y, color = 'black')
plt.show()

##

xm = 0
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        xm += data[i, j]
xm /= n
print("\n\nDыборочное среднее xm = ", xm)
s2 = 0
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        s2 += (data[i, j] - xm) ** 2
s2 /= (n - 1)
print("Выборочная дисперсия s2 = ", s2)

##

plt.figure(figsize=(9,6))
x = binEdgesAverage
y = f1
plt.bar(x, y, width=h)
x1 = np.arange(1,18,0.01)
y1 = scipy.stats.uniform.pdf(x1, loc = 2.598023243281459, scale = 17.93683675671854 - 2.598023243281459)
print(y)
plt.plot(x1, y1, color = 'red')
plt.show()

##

sum = 0
kum =  np.zeros(binEdges[:l+1].size)
for i in range(l):
    sum += p[i]
    kum[i+1] = sum
print(np.around(kum,5), "\n")

##

plt.figure(figsize=(9,6))
x1 = np.arange(0,18,0.01)
y1 = scipy.stats.uniform.cdf(x1, loc = 2.598023243281459, scale = 17.93683675671854 - 2.598023243281459)
plt.plot(x1, y1, color = 'red')
plt.step([0] + binEdgesAverage.tolist()+[18],[0] + kum.tolist())
plt.show()




