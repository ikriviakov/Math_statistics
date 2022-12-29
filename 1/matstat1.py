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
data = [
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
plt.figure(figsize=(10,6))
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

plt.figure(figsize=(10,6))
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

plt.figure(figsize=(10,6))
x1 = np.arange(0,18,0.01)
y1 = scipy.stats.uniform.cdf(x1, loc = 2.598023243281459, scale = 17.93683675671854 - 2.598023243281459)
plt.plot(x1, y1, color = 'red')
plt.step([0] + binEdgesAverage.tolist()+[18],[0] + kum.tolist())
plt.show()












# import numpy as np
# import math
# import scipy
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# def print_matrix(x):
#     for i in range(0, len(x)):
#         for i2 in range(0, len(x[i])):
#             print(x[i][i2], end=' ')
#         print()
# data = [
#     [14.495,4.715,7.175,8.428,11.093,3.375,12.906,8.415,8.916,13.4],
#     [5.343,17.985,15.992,13.89,9.838,13.924,9.012,9.458,17.69,6.542],
#     [14.396,8.592,8.206,14.237,7.357,10.821,12.767,16.058,12.959,4.354],
#     [12.888,10.268,9.182,5.647,8.282,2.903,15.988,12.959,14.919,6.339],
#     [2.375,17.921,9.097,15.85,11.449,11.095,9.493,12.175,7.479,13.535],
#     [9.234,6.078,4.964,6.355,13.957,12.911,15.694,14.286,9.869,5.175],
#     [5.811,7.241,5.814,3.086,6.875,3.878,5.333,15.134,12.924,9.159],
#     [4.727,4.646,15.535,9.919,17.117,10.351,16.892,12.423,10.511,4.942],
#     [4.843,9.927,15.864,3.635,17.963,8.25,5.14,6.734,12.622,13.325],
#     [3.377,16.195,12.04,12.768,2.744,14.186,9.354,15.439,14.612,15.649],
#     [8.681,5.006,3.608,2.867,12.177,15.506,7.683,14.022,17.103,8.905],
#     [12.173,17.757,6.883,2.666,9.861,5.743,16.175,15.308,7.039,15.238]
#     ]
# data = np.array(data, float)
# print("Выборка вариант 9")
# print_matrix(data)




# print("Находим крайние члены вариационного ряда и размах выборки")
# max = np.amax(data)
# min = np.amin(data)
# print("крайние члены вариационного ряда:\nmax = ", max, "min = ", min)
# w = max - min
# print("размах выборки:\nw = ", w)
# n = np.size(data)
# l = math.trunc(1 + math.log2(n))
# h = w / l
# print("размер n = ",n,"\nчисло интервалов l=",l,"ширина интервалов h = ",h)
# histogram,binEdges  = np.histogram(data,l)
# binEdgesAverage = np.zeros(binEdges[:l].size)
# for i in range (l):
#     binEdgesAverage[i] = (binEdges[i] + binEdges[i + 1]) / 2 
# print("средние значения интервалов binEdgesAverage =",binEdgesAverage)
# print("итнтервалы", binEdges)
# p = histogram / n
# print("относительные частоты p = ",np.around(p,5))
# print("частоты histogram = ", histogram)
# f1 = p / h
# print("f1 = ",np.around(f1,7))




# sns.set_theme()
# plt.figure(figsize=(10,6))
# x = binEdgesAverage
# y = f1
# plt.bar(x, y, width=h)
# plt.plot(x, y, color = 'black')
# plt.show()




# xm = 0
# for i in range(data.shape[0]):
#     for j in range(data.shape[1]):
#         xm += data[i, j]
# xm /= n
# print("выборочное среднее xm = ", xm)
# s2 = 0
# for i in range(data.shape[0]):
#     for j in range(data.shape[1]):
#         s2 += (data[i, j] - xm) ** 2
# s2 /= (n - 1)
# print("выборочная дисперсия s2 = ", s2)




# plt.figure(figsize=(10,6))
# x = binEdgesAverage
# y = f1
# plt.bar(x, y, width=h)
# x1 = np.arange(1,18,0.01)
# y1 = scipy.stats.uniform.pdf(x1, loc = 2.598023243281459, scale = 17.93683675671854 - 2.598023243281459)
# print(y)
# plt.plot(x1, y1, color = 'red')
# plt.show()




# sum = 0
# kum =  np.zeros(binEdges[:l+1].size)
# for i in range(l):
#     sum += p[i]
#     kum[i+1] = sum
# print(np.around(kum,5))




# plt.figure(figsize=(10,6))
# x1 = np.arange(0,18,0.01)
# y1 = scipy.stats.uniform.cdf(x1, loc = 2.598023243281459, scale = 17.93683675671854 - 2.598023243281459)
# plt.plot(x1, y1, color = 'red')
# plt.step([0] + binEdgesAverage.tolist()+[18],[0] + kum.tolist())
# plt.show()










































# import pandas as pd
# import numpy as np
# import math
# import matplotlib.pyplot as plt
# import seaborn as sb
# from scipy.stats import expon, uniform
# from prettytable import PrettyTable

# np.set_printoptions(precision=3, suppress=True)


# def print_matrix(x):
#     for i in range(0, len(x)):
#         for i2 in range(0, len(x[i])):
#             print(x[i][i2], end = ' ')
#         print()


# def print_table(th, td):
#     table = PrettyTable(th)
#     columns = len(th) 
#     td_data = td.tolist()[:]
#     while td_data:
#         table.add_row(td_data[:columns])
#         td_data = td_data[columns:]
#     print(table)


# x = [
#     [14.495, 4.715,	7.175,	8.428,	11.093 ,	3.375	, 12.906	, 8.415	    , 8.916	    , 13.48],
#     [5.343,	17.985,	15.992,	13.89,	 9.838 ,	13.924	, 9.012	    , 9.458	    , 17.69	    , 6.542],
#     [14.396, 8.592,	8.206, 14.237,	 7.357 ,	10.821	, 12.767	, 16.058	, 12.959	, 4.354],
#     [12.888, 10.268, 9.182,	5.647,	 8.282 ,	2.903	, 15.988	, 12.959	, 14.919	, 6.339],
#     [2.375,	17.921, 9.097, 15.85,	11.449 ,	11.095	, 9.493	    , 12.175	, 7.479	    , 13.535],
#     [9.234,	6.078,	4.964,	6.355,	13.957 ,	12.911	, 15.694	, 14.286	, 9.869	    , 5.175],
#     [5.811,	7.241,	5.814,	3.086,	 6.875 ,	3.878	, 5.333	    , 15.134	, 12.924	, 9.159],
#     [4.727,	4.646,	15.535,	9.919,	17.117 ,	10.351	, 16.892	, 12.423	, 10.511	, 4.942] ,
#     [4.843,	9.927,	15.864,	3.635,	17.963 ,	8.25	, 5.14	    , 6.734	    , 12.622	, 13.325],
#     [3.377,	16.195,	12.04, 12.768,	 2.744 ,	14.186	, 9.354	    , 5.439	    , 14.612	, 15.649],
#     [8.681,	5.006,	3.608,	2.867,	12.177 ,	15.506	, 7.683	    , 14.022	, 17.103	, 8.905],
#     [12.173, 17.757, 6.883,	2.666,	 9.861 ,	5.743	, 16.175	, 15.308	, 7.039	    , 15.238],
# ]


# x = np.array(x, float)
# print("\n\nВыборка (вариант 9) \n", x)


# n = x.size
# Max = np.amax(x)
# Min = np.amin(x)
# w = Max - Min

# print("\n\nРазмер n = %d\nКрайние члены вариационного ряда max = %.3f, min = %.5f\nРазмах выборки w = %f" 
#       %(n, Max, Min, w))


# l = math.trunc(1 + math.log2(n))
# h = w / l

# print("\n\nЧисло интервалов l = %d\nШаг(ширина интервалов группировки) h = %f" %(l, h))

# hist, bin_edges = np.histogram(x, l)
# bin_edges_average = np.zeros(bin_edges[:l].size)
# for i in range (l):
#     bin_edges_average[i] = (bin_edges[i] + bin_edges[i + 1]) / 2 
# f1 = bin_edges_average
# f2 = hist

# print("\n")
# th = ["\nCредние значения каждого из  интервалов", "Частоты попадания элементов в каждый из интервалов"]
# td = np.array(f1[0])
# for i in range(len(f1)-1):
#     td = np.append(td, f2[i])
#     td = np.append(td, f1[i+1])
# td = np.append(td, f2[6])
# print_table(th, td)

# p = hist / n

# Int = f1
# intl = Int - h / 2
# intl = np.append(intl, Max)

# print("\n")
# th = ["\nИнтервал", "Середина интервала int", "Частота f2", "Относительная частота p"]
# td = []
# for i in range(l):
#     td.append("["+str(round(intl[i], 4))+","+str(round(intl[i + 1], 4))+")")
#     td.append(round(Int[i], 4))
#     td.append(f2[i])
#     td.append(round(p[i], 4))
# print_table(th, np.array(td))

# sb.set_style("whitegrid")
# plt.figure(figsize=(10,6))
# X = Int
# Y = p / h
# plt.bar(X, Y, width=h, color='navy')
# plt.plot(X, Y, color='red')
# plt.xlabel("int")
# plt.ylabel("p / h")
# plt.show()





# xm = sum(x[i, j] for i in range(np.shape(x)[0]) for j in range(np.shape(x)[1])) / n
# s2 = sum((x[i, j] - xm) ** 2 for i in range(np.shape(x)[0]) for j in range(np.shape(x)[1])) / (n - 1)
# print("\nВыброчное среднее xm =", round(xm, 5))
# print("\nВыборочная дисперсия s2 =", round(s2, 5))

# sb.set_style("whitegrid")
# ls = np.linspace(2.6, 17.9)
# pdf = uniform.pdf(ls, scale = xm)
# plt.plot(ls, pdf, color='red')
# plt.xlabel('Гистограмма и график плотности показательного закона')
# X = Int
# Y = p / h
# plt.bar(X, Y, width=h, color='navy')
# plt.show()




# def ind(x):
#     if (x > 0).any():
#         return 1
#     return 0

# kum = np.empty(len(p) + 1)
# kum[0] = 0
# l = len(Int) - 1
# for k in range(1, l + 2):
#     kum[k] = sum(p[i] for i in range(0, k))
    
# def femp(x):
#     return sum(p[i] * ind(x - Int[i]) for i in range(l))
# print("\nkum = ", kum)

# sb.set_style("whitegrid")
# ls = np.linspace(3, 18)
# cdf = uniform.cdf(ls, scale = xm)
# plt.plot(ls, cdf, color='red')
# plt.plot(ls, np.ones(len(ls)), color='black')
# plt.step([0] + Int.tolist() + [18], [0] + kum.tolist(), color='navy')
# plt.xlabel('z')
# plt.ylabel('Черный:1\nКрасный:Теоретическая\nСиний:Эмперическая')
# plt.show()


















































































































































# from ctypes.wintypes import HINSTANCE
# from this import d
# import numpy as np 
# import math 
# import scipy
# import pandas as pd 

# def print_matrix(x):
#     for i in range(0, len(x)):
#         for i2 in range(0, len(x[i])):
#             print(x[i][i2], end = ' ')
#         print()

# data = [
#     [14.495, 4.715,	7.175,	8.428,	11.093 ,	3.375	, 12.906	, 8.415	    , 8.916	    , 13.48],
#     [5.343,	17.985,	15.992,	13.89,	 9.838 ,	13.924	, 9.012	    , 9.458	    , 17.69	    , 6.542],
#     [14.396, 8.592,	8.206, 14.237,	 7.357 ,	10.821	, 12.767	, 16.058	, 12.959	, 4.354],
#     [12.888, 10.268, 9.182,	5.647,	 8.282 ,	2.903	, 15.988	, 12.959	, 14.919	, 6.339],
#     [2.375,	17.921, 9.097, 15.85,	11.449 ,	11.095	, 9.493	    , 12.175	, 7.479	    , 13.535],
#     [9.234,	6.078,	4.964,	6.355,	13.957 ,	12.911	, 15.694	, 14.286	, 9.869	    , 5.175],
#     [5.811,	7.241,	5.814,	3.086,	 6.875 ,	3.878	, 5.333	    , 15.134	, 12.924	, 9.159],
#     [4.727,	4.646,	15.535,	9.919,	17.117 ,	10.351	, 16.892	, 12.423	, 10.511	, 4.942] ,
#     [4.843,	9.927,	15.864,	3.635,	17.963 ,	8.25	, 5.14	    , 6.734	    , 12.622	, 13.325],
#     [3.377,	16.195,	12.04, 12.768,	 2.744 ,	14.186	, 9.354	    , 5.439	    , 14.612	, 15.649],
#     [8.681,	5.006,	3.608,	2.867,	12.177 ,	15.506	, 7.683	    , 14.022	, 17.103	, 8.905],
#     [12.173, 17.757, 6.883,	2.666,	 9.861 ,	5.743	, 16.175	, 15.308	, 7.039	    , 15.238],
# ]


# data = np.array(data, float)
# print("\nВыборка Вариант 9")
# print_matrix(data)





# print("\n\nНаходим крайние члены вариационного рода и размах выборки")
# max = np.amax(data)
# min = np.amin(data)
# print("Крайние члены вариационного ряда:\nmax = ", max, "min = ", min)
# w = max - min
# print("Размах выборки:\nw = ", w)

# n = np.size(data)
# l = math.trunc(1 + math.log2(n))
# h = w/1
# print("\n\nРазмер n = ", n, "\nЧисло интервалов l = ", l, "\nШирина Интервалов h = ", h, "\n\n")




# histogram, binEDdges = np.histogram(data, l)
# binEDdgesAverage = np.zeros(binEDdges[:l].size)
# for i in range (l):
#     binEDdgesAverage[i] = (binEDdges[i] + binEDdges[i + 1]) / 2
#     print("\nСредние значения интервалов binEDdgesAverage = ", binEDdgesAverage)

# xm = 0
# for i in range(data.shape[0]):
#     for j in range(data.shape[1]):
#         xm += data[i, j]

# xm /= n
# print("\n\nВыборочное среднее xm = ", xm)

# s2 = 0
# for i in range(data.shape[0]):
#     for j in range(data.shape[1]):
#         s2 += (data[i, j] - xm) ** 2
# s2 /= (n - 1)
# print("Выборочная дисперсия s2 = ", s2)


# sum = 0
# kum = np.zeros(binEDdges[:l + 1].size)
# for i in range(l):
#     sum += p[i]
#     kum[i+1] = sum
# print(np.around(kum, 5))







