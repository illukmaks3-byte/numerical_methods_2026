import numpy as np

LAB_NUMBER = 4

# =========================
# ФУНКЦІЇ
# =========================
def M(t):
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)

def dM_exact(t):
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)

def dM_numeric(t, h):
    return (M(t + h) - M(t - h)) / (2 * h)

# =========================
# ОСНОВНА ПРОГРАМА
# =========================
def main():
    print("=" * 60)
    print(f"ЛАБОРАТОРНА РОБОТА №{4}")
    print("Чисельне диференціювання")
    print("=" * 60)

    # Ввід
    t0 = float(input("Введіть t (наприклад 1): "))

    # -------------------------
    # 1. АНАЛІТИЧНЕ
    # -------------------------
    exact = dM_exact(t0)

    print("\n1. АНАЛІТИЧНЕ ЗНАЧЕННЯ")
    print(f"M'(t) = {exact:.10f}")

    # -------------------------
    # 2. ДОСЛІДЖЕННЯ ПОХИБКИ
    # -------------------------
    print("\n2. ДОСЛІДЖЕННЯ ПОХИБКИ")
    print(f"{'h':<15}{'D(h)':<20}{'Похибка':<20}")

    h_values = np.logspace(-10, 1, 50)

    best_h = None
    min_error = float("inf")
    best_D = None

    for h in h_values:
        d = dM_numeric(t0, h)
        error = abs(d - exact)

        print(f"{h:<15.2e}{d:<20.10f}{error:<20.10e}")

        if error < min_error:
            min_error = error
            best_h = h
            best_D = d

    print("\nОптимальний крок h0 =", best_h)
    print("Досягнута точність R0 =", min_error)

    # -------------------------
    # 3. ПРИЙМАЄМО h
    # -------------------------
    h = best_h

    # -------------------------
    # 4. ДВА КРОКИ
    # -------------------------
    d_h = dM_numeric(t0, h)
    d_h2 = dM_numeric(t0, h / 2)

    print("\n3. ДВА КРОКИ")
    print(f"D(h)   = {d_h:.10f}")
    print(f"D(h/2) = {d_h2:.10f}")

    # -------------------------
    # 5. ПОХИБКА
    # -------------------------
    print("\n4. ПОХИБКА")
    print(f"ε(h) = {abs(d_h - exact):.10e}")

    # -------------------------
    # 6. РУНГЕ-РОМБЕРГ
    # -------------------------
    p = 2
    runge = d_h2 + (d_h2 - d_h) / (2**p - 1)

    print("\n5. МЕТОД РУНГЕ-РОМБЕРГА")
    print(f"Уточнене значення = {runge:.10f}")
    print(f"Похибка = {abs(runge - exact):.10e}")

    # -------------------------
    # 7. ЕЙТКЕН
    # -------------------------
    d_h4 = dM_numeric(t0, h / 4)

    try:
        aitken = d_h - ((d_h2 - d_h)**2) / (d_h4 - 2*d_h2 + d_h)
    except ZeroDivisionError:
        aitken = d_h

    print("\n6. МЕТОД ЕЙТКЕНА")
    print(f"D(h)   = {d_h:.10f}")
    print(f"D(h/2) = {d_h2:.10f}")
    print(f"D(h/4) = {d_h4:.10f}")
    print(f"Уточнене значення = {aitken:.10f}")
    print(f"Похибка = {abs(aitken - exact):.10e}")

    # -------------------------
    # ВИСНОВОК
    # -------------------------
    print("\n7. ВИСНОВОК")
    print(f"Точне значення        = {exact:.10f}")
    print(f"Оптимальний крок h0   = {best_h:.2e}")
    print(f"Рунге-Ромберг         = {runge:.10f}")
    print(f"Ейткен (найточніше)   = {aitken:.10f}")

    print("=" * 60)


# =========================
# ЗАПУСК
# =========================
if __name__ == "__main__":
    main()