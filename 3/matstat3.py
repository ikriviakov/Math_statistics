import numpy as np
import math
import scipy
import pandas as pd
import random
import seaborn as sb
import matplotlib.pyplot as plt

n = 120

##

def p(x):
    if x <= 0:
        return 0
    else:
        coef = (1 / (np.sqrt(0.4 * np.pi) * x))
        exp_coef = - (np.log(x) - 2) ** 2 / 0.4
        
        return coef * np.exp(exp_coef)

def F(x):
    if x <= 0:
        return 0
    else: 
        return 0.5 * (math.erf((np.log(x) - 2) / (np.sqrt(0.4))) +1 )
    
def obrF(y):
    if y <=0:
        return 0
    else:
        return math.exp(2 + np.sqrt(0.4) * scipy.special.erfinv(2 * y - 1))

def funToArray(array, fun):
    arr = []
    for x in array:
        arr.append(fun(x))
    return arr

def ind(x):
    if (x > 0):
        return 1
    return 0

def Femp(z):
    result = 0
    for i in X:
        result += ind(z - i)
    result /= n
    return result
    
hama = 0.1
eps = math.sqrt(-1/(2 * n) * math.log(hama / 2))

def R(z):
    value = Femp(z) + eps 
    if value > 1:
        return 1
    else:
        return value

def L(z):
    value = Femp(z) - eps
    return Femp(z) - eps
    if value > 0:
        return value
    else:
        return 0

##

x = np.arange(0, 35, 0.001)
sb.set_style("whitegrid")
plt.plot(x, funToArray(x,p), color='navy')
plt.xlabel('Плотность распределения логнормального закона')
plt.show()

##

Y = np.array([])
for i in range(n):
    Y = np.append(Y, random.random())
print("Моделируем массив из 120 случайных чисел:\n", Y)
print("")

print("Проходимся по смоделированному массиву и формируем новый,\nэлементами которого будут значения обратной функции распредления")

X = np.array([])
X = np.append(X, funToArray(Y,obrF))
print("\nСмоделированный массив X:")
print(X)

##

print("\n\nНаходим крайние члены вариационного ряда и размах выборки")
max = np.amax(X)
min = np.amin(X)
print("Крайние члены вариационного ряда:\nmax = ", max, "\nmin = ", min)
w = max - min
print("Размах выборки:\nw = ", w)

n = np.size(X)
l = math.trunc(1 + math.log2(n))
h = w / l
print("Размер n = ",n,"\nЧисло интервалов l = ",l,"\nШирина интервалов h = ",h)

histogram,binEdges  = np.histogram(X,l)
binEdgesAverage = np.zeros(binEdges[:l].size)
for i in range (l):
    binEdgesAverage[i] = (binEdges[i] + binEdges[i + 1]) / 2 
print("\n\nСредние значения интервалов binEdgesAverage:\n",binEdgesAverage)
print("Итнтервалы:\n", binEdges)
rf = histogram / n
print("Относительные частоты rf:\n",np.around(rf,5))
print("Частоты histogram:\n", histogram)
f1 = rf / h
print("f1\n",np.around(f1,7))

##

sb.set_theme()
plt.figure(figsize=(10,6))
x = binEdgesAverage
y = f1
plt.bar(x, y, width=h)
plt.plot(x, y, color = 'black')
plt.show()

##

xm = 0
for i in range(X.shape[0]):
    xm += X[i]
xm /= n
print("\n\nВыборочное среднее xm = ", xm)

s2 = 0
for i in range(X.shape[0]):
    s2 += (X[i] - xm) ** 2
s2 /= (n - 1)
print("Выборочная дисперсия s2 = ", s2)

##

plt.figure(figsize=(10,6))
x = binEdgesAverage
y = f1
plt.bar(x, y, width=h)
x1 = np.arange(1,20.4,0.01)
y1 =  funToArray(x1,p)
plt.plot(x1, y1, color = 'red')
plt.show()

##

m1 = np.exp(21 / 10)
m2 = np.exp(22 / 5)
d = m2 - m1**2
print("Матожидание m1 = ",m1,"\nДисперсия d = ",d)
print("Эмпирическое среднее xm = ",xm,"\nВыборочная дисперсия s2 = ",s2)
print("Сравнение m1 - xm = ",m1 - xm,"\nd / s2 = ", d / s2, "\n")

##

print("\nСмоделированный массив из 120 случайных чисел:\n", Y)

##

sb.set_style("whitegrid")

x = np.arange(0, 30, 0.001)
plt.figure(figsize=(10,6))
plt.plot(x, funToArray(x, F), color='navy')
plt.plot(x, funToArray(x, Femp), color='red')
plt.plot(x, funToArray(x, R), color='green')
plt.plot(x, funToArray(x, L), color='black')
plt.xlabel("z")
plt.ylabel("Синий:F(z)\nКрасный:Femp(z)\nЗеленый:R(z)\nЧерный:L(z)\n")
plt.show()

##



# s3 = 0
# for i in range(X.shape[0]):
#     s3 += (X[i]) ** 2
# s3 /= (n)
# print("Выборочная дисперся без выборочного среднего s3 = ", s3)

# a = ((4 * math.log(xm)) - math.log(s3)) / 2
# print("\n\nПараметр a = ", a)

# sigma = math.sqrt(math.log(s3) - (4 * math.log(xm)))
# print("Параметр sigma^2 = ", sigma)


# l1 = math.sqrt(0.4) / math.sqrt(2)
# print(l1)
# # xm - выборочное среднее
# # s2 - выборочная дисперсия