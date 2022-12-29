import numpy as np
import math
import random
import seaborn as sb
import matplotlib.pyplot as plt
from prettytable import PrettyTable

np.set_printoptions(precision=7, suppress=True)

def print_table(th, td):
    table = PrettyTable(th)
    columns = len(th) 
    td_data = td.tolist()[:]
    while td_data:
        table.add_row(td_data[:columns])
        td_data = td_data[columns:]
    print(table)

k = 8
p = 0.7
n = 140

print("Вариант 9\nk =", k, "\np =", p, "\nn =", n)

P = np.array([])
for i in range(k + 1):
    cur = math.comb(k, i) * p ** i * (1 - p) ** (k - i)
    P = np.append(P, cur)
    
U = np.array([])
s = 0
for i in P:
    s += i
    U = np.append(U, s)

print("Находим теоретический закон\n")
th = ['Значение СВ', 0, 1, 2, 3, 4, 5, 6, 7, 8]
td = ['Вероятности']
td = np.append(td, np.around(P, 5))
td = np.append(td, "Кумулятивные вер-ти")
td = np.append(td, np.around(U, 5))
print_table(th, td)
print("\n")
print("Вероятности вычисляются по формуле Бернулли:\n p =", P)
print("\n")
print("Кумулятивные вероятности:\n u =", U)

##

y = np.array([])
for i in range(n):
    y = np.append(y, random.random())

x = np.array([])
for y_n in y:
    for i in range(U.size):
        if y_n < U[i]:
            x = np.append(x, i)
            break
            
frequency = np.array([])
for i in range(k + 1):
    frequency = np.append(frequency, np.count_nonzero(x == i))
    
relative_frequency = np.array([])
for i in range(frequency.size):
    relative_frequency = np.append(relative_frequency, frequency[i] / n) 
    
accumulated_frequency = np.array([])
s = 0
for i in relative_frequency:
    s += i
    accumulated_frequency = np.append(accumulated_frequency, s)
    
print("Моделируем вектор из n случайных чисел\n")
print("y = ", y)
print("\n")
print("По вектору y разыгрываем вектор x в соответствии с алгоритмом\n")
print("x = ", x.astype(int))
print("\n")
print("Строим статистический ряд (здесь n=140)\n")
th = ['Значение СВ', 0, 1, 2, 3, 4, 5, 6, 7, 8]
td = ['Частоты']
td = np.append(td, frequency)
td = np.append(td, ['Относительные частоты'])
td = np.append(td, np.around(relative_frequency, 3))
td = np.append(td, ['Накопительные частоты'])
td = np.append(td, np.around(accumulated_frequency, 3))

print_table(th, td)

##

print("Строим совмещенные графики")
sb.set_style('whitegrid')
z = range(0, 10)
plt.step(z, [0.0] + U.tolist(), linewidth = 1.0, color='black')
plt.step(z, [0.0] + accumulated_frequency.tolist(), linewidth = 1.0, color='red')
plt.xlabel("z")
plt.ylabel("Черный: F(z)\nКрасный: F14e0(z)")
plt.show()

##

modul = np.array([0])
for i in range(k + 1):
    modul = np.append(modul, abs(accumulated_frequency[i] - U[i]))
    
print("Вычисление статистики Колмогорова для выборки из дискретного закона\n")
th = ['Интервалы', '(-inf,0]', '(0,1]', '(1,2]', '(2,3]', '(3,4]', '(4,5]', '(5,6]', '(6,7]', '(7,8]', '(8,+inf)']
td = ['Эмпирическая\nфункция\nраспределения\n', 0]
td = np.append(td, np.around(accumulated_frequency,4))
td = np.append(td, ['Теоретическая\nфункция\nраспределения\n'])
td = np.append(td, 0)
td = np.append(td, np.around(U, 4))
td = np.append(td, ['Модуль\nразности\n'])
td = np.append(td, np.around(modul, 4))

print_table(th, td)

print("Статистика Колмагорова:\n", np.amax(modul))

##

print("Вычисление эмпирических и теоретических характеристик\n")

mn = x.mean()
m = sum(P[i] * i for i in range(k + 1))
D = sum((i - m) ** 2 * P[i] for i in range(k + 1))
S2 = 1/179 * sum((x[i] - mn) ** 2 for i in range(n))

print("Абсолютная величина разности выборочного среднего и теоретического мат ожидания мало\n", np.around(abs(mn-m), 5))
print("Отношение среднеквадратичных отклонений близко к единице\n", np.around(math.sqrt(S2/D), 5))
print("Следовательно, результаты моделирования можно признать удовлетворительными.")

