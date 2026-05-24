import math
import numpy as np
import matplotlib.pyplot as plt

# =========================================
# ФУНКЦІЯ
# =========================================
def f(x):
    return x**3 - 4*x - 1

def df(x):
    return 3*x**2 - 4

def d2f(x):
    return 6*x


# =========================================
# ТАБУЛЯЦІЯ ФУНКЦІЇ
# =========================================
def tabulate(a, b, h):
    x = a

    file = open("table.txt", "w")

    while x <= b:
        y = f(x)
        file.write(f"{x:.4f}  {y:.4f}\n")
        print(f"x = {x:.4f}  y = {y:.4f}")
        x += h

    file.close()


# =========================================
# МЕТОД ПРОСТОЇ ІТЕРАЦІЇ
# =========================================
def simple_iteration(x0, eps, max_iter=100):

    def phi(x):
        return (x**3 - 1) / 4

    x = x0
    k = 0

    while k < max_iter:
        x_new = phi(x)

        if abs(f(x_new)) < eps and abs(x_new - x) < eps:
            return x_new, k + 1

        x = x_new
        k += 1

    return x, k


# =========================================
# МЕТОД НЬЮТОНА
# =========================================
def newton(x0, eps, max_iter=100):
    x = x0
    k = 0

    while k < max_iter:
        x_new = x - f(x) / df(x)

        if abs(f(x_new)) < eps and abs(x_new - x) < eps:
            return x_new, k + 1

        x = x_new
        k += 1

    return x, k


# =========================================
# МЕТОД ЧЕБИШЕВА
# =========================================
def chebyshev(x0, eps, max_iter=100):
    x = x0
    k = 0

    while k < max_iter:
        fx = f(x)
        dfx = df(x)
        d2fx = d2f(x)

        x_new = x - fx/dfx - (fx**2 * d2fx) / (2 * dfx**3)

        if abs(f(x_new)) < eps and abs(x_new - x) < eps:
            return x_new, k + 1

        x = x_new
        k += 1

    return x, k


# =========================================
# МЕТОД ХОРД
# =========================================
def chord(a, b, eps, max_iter=100):
    x0 = a
    x1 = b
    k = 0

    while k < max_iter:
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))

        if abs(f(x2)) < eps and abs(x2 - x1) < eps:
            return x2, k + 1

        x0 = x1
        x1 = x2
        k += 1

    return x2, k


# =========================================
# МЕТОД ПАРАБОЛ
# =========================================
def parabola(x0, x1, x2, eps, max_iter=100):
    k = 0

    while k < max_iter:

        f0 = f(x0)
        f1 = f(x1)
        f2 = f(x2)

        h1 = x1 - x0
        h2 = x2 - x1

        d1 = (f1 - f0) / h1
        d2 = (f2 - f1) / h2

        a = (d2 - d1) / (h2 + h1)
        b = a * h2 + d2
        c = f2

        D = math.sqrt(b*b - 4*a*c)

        if abs(b + D) > abs(b - D):
            den = b + D
        else:
            den = b - D

        dx = -2*c / den
        x3 = x2 + dx

        if abs(f(x3)) < eps and abs(dx) < eps:
            return x3, k + 1

        x0 = x1
        x1 = x2
        x2 = x3

        k += 1

    return x3, k


# =========================================
# ЗВОРОТНА ІНТЕРПОЛЯЦІЯ
# =========================================
def inverse_interpolation(x0, x1, eps, max_iter=100):
    k = 0

    while k < max_iter:

        f0 = f(x0)
        f1 = f(x1)

        x2 = (x0*f1 - x1*f0) / (f1 - f0)

        if abs(f(x2)) < eps and abs(x2 - x1) < eps:
            return x2, k + 1

        x0 = x1
        x1 = x2

        k += 1

    return x2, k


# =========================================
# ГРАФІК
# =========================================
def draw_graph():

    x = np.linspace(-5, 5, 500)
    y = [f(i) for i in x]

    plt.figure(figsize=(8, 5))
    plt.plot(x, y)
    plt.axhline(0)
    plt.axvline(0)

    plt.grid()
    plt.title("Графік функції")
    plt.xlabel("x")
    plt.ylabel("f(x)")

    plt.show()


# =========================================
# ОСНОВНА ПРОГРАМА
# =========================================
eps = 1e-6

print("ТАБУЛЯЦІЯ")
tabulate(-5, 5, 0.5)

print("\nГРАФІК")
draw_graph()

print("\nМЕТОД НЬЮТОНА")
root, it = newton(2, eps)
print("Корінь =", root)
print("Ітерацій =", it)

print("\nМЕТОД ПРОСТОЇ ІТЕРАЦІЇ")
root, it = simple_iteration(1, eps)
print("Корінь =", root)
print("Ітерацій =", it)

print("\nМЕТОД ЧЕБИШЕВА")
root, it = chebyshev(2, eps)
print("Корінь =", root)
print("Ітерацій =", it)

print("\nМЕТОД ХОРД")
root, it = chord(1, 3, eps)
print("Корінь =", root)
print("Ітерацій =", it)

print("\nМЕТОД ПАРАБОЛ")
root, it = parabola(0, 1, 2, eps)
print("Корінь =", root)
print("Ітерацій =", it)

print("\nЗВОРОТНА ІНТЕРПОЛЯЦІЯ")
root, it = inverse_interpolation(1, 2, eps)
print("Корінь =", root)
print("Ітерацій =", it)