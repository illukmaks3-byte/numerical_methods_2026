import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# ВХІДНІ ДАНІ ТА ТЕСТОВА ЗАДАЧА КОШІ (Варіант для демонстрації)
# =====================================================================
def f(x, y):
    """Права частина диференціального рівняння dy/dx = f(x, y)"""
    return y - x**2 + 1

def exact_sol(x):
    """Аналітичний (точний) розв'язок рівняння для порівняння похибок"""
    return (x + 1)**2 - 0.5 * np.exp(x)

# Параметри інтегрування (Завдання 1, 2, 6)
x0, xN = 0.0, 2.0
y0 = 0.5
h_fixed = 0.1
eps = 1e-5  # Задана точність для автоматичного кроку

# =====================================================================
# ЧАСТИНА 2: МЕТОД РУНГЕ-КУТТА 4-ГО ПОРЯДКУ (РК4)
# =====================================================================

def rk4_step(x, y, h):
    """Один крок методом Рунге-Кутта 4-го порядку"""
    k1 = f(x, y)
    k2 = f(x + h/2, y + h*k1/2)
    k3 = f(x + h/2, y + h*k2/2)
    k4 = f(x + h, y + h*k3)
    return y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

def rk4_fixed(x0, xN, y0, h):
    """РК4 із фіксованим кроком (Завдання 6)"""
    X = np.arange(x0, xN + h/2, h)
    Y = np.zeros(len(X))
    Y[0] = y0
    for i in range(len(X) - 1):
        Y[i+1] = rk4_step(X[i], Y[i], h)
    return X, Y

def rk4_adaptive(x0, xN, y0, eps):
    """РК4 з автоматичним вибором кроку за методом Рунге (Завдання 8, 9)"""
    X, Y, H = [x0], [y0], []
    x, y, h = x0, y0, 0.1  # початковий орієнтовний крок
    
    while x < xN:
        if x + h > xN:
            h = xN - x
            
        # Один крок довжиною h
        y_h = rk4_step(x, y, h)
        # Два кроки довжиною h/2
        y_h2 = rk4_step(x, y, h/2)
        y_h2 = rk4_step(x + h/2, y_h2, h/2)
        
        # Оцінка похибки за методом Рунге (Завдання 8)
        err = (16.0 / 15.0) * abs(y_h - y_h2)
        
        # Критерій прийняття кроку (Завдання 9)
        if err > eps:
            h /= 2  # Крок завеликий, зменшуємо
        else:
            # Крок прийнято
            x += h
            y = y_h2  # Уточнене значення за Рунге
            X.append(x)
            Y.append(y)
            H.append(h)
            
            # Чи потрібно збільшити крок для наступної ітерації?
            if err < eps / 32:  # Константа k = 2^(4+1) = 32
                h *= 2
                
    return np.array(X), np.array(Y), np.array(H)

# =====================================================================
# ЧАСТИНА 1: МЕТОД ПРОГНОЗУ ТА КОРЕКЦІЇ АДАМСА 2-ГО ПОРЯДКУ
# =====================================================================

def adams_fixed(x0, xN, y0, h, max_iter=10):
    """Метод прогнозу-корекції Адамса 2-го порядку з фіксованим кроком (Завдання 2)"""
    X = np.arange(x0, xN + h/2, h)
    Y = np.zeros(len(X))
    Y[0] = y0
    
    # Стартовий крок за допомогою РК4 (оскільки метод 2-кроковий)
    Y[1] = rk4_step(X[0], Y[0], h)
    
    err_theoretical = [0.0, 0.0] # Для перших точок оцінка Адамса не рахується
    
    for n in range(1, len(X) - 1):
        f_n = f(X[n], Y[n])
        f_nm1 = f(X[n-1], Y[n-1])
        
        # Етап прогнозу (Предиктор)
        y_pred = Y[n] + (h / 2) * (3 * f_n - f_nm1)
        
        # Етап корекції (Ітераційний коректор)
        y_corr = y_pred
        for _ in range(max_iter):
            y_corr_next = Y[n] + (h / 2) * (f(X[n+1], y_corr) + f_n)
            if abs(y_corr_next - y_corr) < 1e-7:
                y_corr = y_corr_next
                break
            y_corr = y_corr_next
            
        # Теоретична оцінка локальної похибки на етапі корекції (Завдання 4)
        # R_corr ≈ -(1/6) * |y_corr - y_pred| (згідно з виведенням похибок)
        err_theoretical.append(abs(y_corr - y_pred) / 6.0)
        
        Y[n+1] = y_corr
        
    return X, Y, err_theoretical

def adams_adaptive(x0, xN, y0, eps):
    """Метод Адамса 2-го порядку з автоматичним вибором кроку (Завдання 5)"""
    X, Y, H = [x0], [y0], []
    x, y, h = x0, y0, 0.1
    
    # Оскільки крок змінюється динамічно, найпростіший та найнадійніший спосіб 
    # адаптації Адамса — перезапуск РК4 для отримання передісторії на кожному етапі.
    while x < xN:
        if x + h > xN:
            h = xN - x
            
        # Отримуємо додаткову точку через РК4
        x1 = x + h
        y1_rk = rk4_step(x, y, h)
        
        # Робимо крок Адамса
        f_n = f(x, y)
        f_n1 = f(x1, y1_rk)
        y2_pred = y1_rk + (h / 2) * (3 * f_n1 - f_n)
        
        x2 = x + 2*h
        y2_corr = y2_pred
        for _ in range(5):
            y2_corr_next = y1_rk + (h / 2) * (f(x2, y2_corr) + f_n1)
            if abs(y2_corr_next - y2_corr) < 1e-8:
                y2_corr = y2_corr_next
                break
            y2_corr = y2_corr_next
            
        # Оцінка локальної похибки Адамса
        err = abs(y2_corr - y2_pred) / 6.0
        
        if err > eps:
            h /= 2  # Зменшуємо крок
        else:
            # Приймаємо крок (рухаємось на 2х за раз для зручності передісторії)
            X.append(x1)
            Y.append(y1_rk)
            H.append(h)
            
            if x2 <= xN:
                X.append(x2)
                Y.append(y2_corr)
                H.append(h)
                x = x2
                y = y2_corr
            else:
                x = x1
                y = y1_rk
                
            if err < eps / 8: # Для 2-го порядку константа k = 2^(2+1) = 8
                h *= 2
                
    return np.array(X[:len(H)+1]), np.array(Y[:len(H)+1]), np.array(H)

# =====================================================================
# ОБЧИСЛЕННЯ ТА ВИНЕСЕННЯ РЕЗУЛЬТАТІВ
# =====================================================================

# 1. Розрахунки з фіксованим кроком
X_adams, Y_adams, err_adams_theory = adams_fixed(x0, xN, y0, h_fixed)
X_rk4, Y_rk4 = rk4_fixed(x0, xN, y0, h_fixed)

Y_exact_adams = exact_sol(X_adams)
Y_exact_rk4 = exact_sol(X_rk4)

# Справжні локальні похибки порівняно з аналітичним розв'язком (Завдання 3, 7)
err_adams_exact = abs(Y_adams - Y_exact_adams)
err_rk4_exact = abs(Y_rk4 - Y_exact_rk4)

# 2. Розрахунки з автоматичним кроком
X_adams_adp, Y_adams_adp, H_adams_adp = adams_adaptive(x0, xN, y0, eps)
X_rk4_adp, Y_rk4_adp, H_rk4_adp = rk4_adaptive(x0, xN, y0, eps)

# =====================================================================
# ВІЗУАЛІЗАЦІЯ (ПОБУДОВА ГРАФІКІВ)
# =====================================================================
plt.figure(figsize=(14, 10))

# Графік 1: Порівняння чисельних методів із точним розв'язком
plt.subplot(2, 2, 1)
x_fine = np.linspace(x0, xN, 200)
plt.plot(x_fine, exact_sol(x_fine), 'g-', label='Точний розв\'язок', linewidth=2)
plt.plot(X_adams, Y_adams, 'ro--', label='Адамс (фiкс. крок)', alpha=0.7)
plt.plot(X_rk4, Y_rk4, 'bs--', label='Рунге-Кутта 4 (фiкс. крок)', alpha=0.7)
plt.title('Порівняння знайдених розв\'язків')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

# Графік 2: Локальна похибка Адамса (Завдання 3 та 4)
plt.subplot(2, 2, 2)
plt.plot(X_adams, err_adams_exact, 'r-o', label='Справжня похибка (|Y_чисел - Y_точне|)')
# Додаємо теоретичну оцінку (починаючи з 3-ї точки, де вона визначена)
plt.plot(X_adams[2:], err_adams_theory[2:], 'k--', label='Теоретична оцінка (Предиктор-Коректор)')
plt.title('Частина 1: Локальна похибка методу Адамса')
plt.xlabel('x')
plt.ylabel('Похибка')
plt.legend()
plt.grid(True)

# Графік 3: Локальна похибка РК4 (Завдання 7)
plt.subplot(2, 2, 3)
plt.plot(X_rk4, err_rk4_exact, 'b-s', label='Справжня похибка РК4')
plt.title('Частина 2: Локальна похибка РК4')
plt.xlabel('x')
plt.ylabel('Похибка')
plt.legend()
plt.grid(True)

# Графік 4: Автоматичний вибір кроку (Завдання 5 та 9)
plt.subplot(2, 2, 4)
plt.step(X_adams_adp[:-1], H_adams_adp, 'r-+', where='post', label='Крок Адамса')
plt.step(X_rk4_adp[:-1], H_rk4_adp, 'b-x', where='post', label='Крок РК4')
plt.title(f'Динаміка автоматичного кроку (eps = {eps})')
plt.xlabel('x')
plt.ylabel('Величина кроку h')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
