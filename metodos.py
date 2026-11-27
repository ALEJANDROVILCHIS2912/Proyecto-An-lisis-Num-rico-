"""Métodos numéricos implementados desde cero (solo NumPy).

Contenido
---------
* EDOs: método de Euler y Runge-Kutta de 4.º orden (RK4).
* Sistemas no lineales: Newton-Raphson con Jacobiano analítico.
* Modelos mecánicos: péndulo simple, péndulo doble y mecanismo de cuatro barras.
"""
import numpy as np

GRAVEDAD = 9.81  # m/s²


# ---------------------------------------------------------------------------
# 1. Integración de EDOs:  y' = f(t, y)
# ---------------------------------------------------------------------------
def euler_paso(f, t, y, h):
    """Un paso del método de Euler explícito (orden 1)."""
    return y + h * f(t, y)


def rk4_paso(f, t, y, h):
    """Un paso de Runge-Kutta clásico de 4.º orden."""
    k1 = f(t, y)
    k2 = f(t + h / 2, y + h / 2 * k1)
    k3 = f(t + h / 2, y + h / 2 * k2)
    k4 = f(t + h, y + h * k3)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def integrar(f, y0, tf, h, metodo="rk4"):
    """Integra y' = f(t, y) desde t = 0 hasta tf con paso fijo h.

    Devuelve (t, Y) con Y de forma (n+1, len(y0)).
    """
    paso = rk4_paso if metodo == "rk4" else euler_paso
    n = int(round(tf / h))
    t = np.arange(n + 1) * h
    Y = np.empty((n + 1, len(y0)))
    Y[0] = y0
    for i in range(n):
        Y[i + 1] = paso(f, t[i], Y[i], h)
    return t, Y


# ---------------------------------------------------------------------------
# 2. Modelos de péndulo
# ---------------------------------------------------------------------------
def pendulo_simple(L):
    """Estado y = [theta, omega].   theta'' = -(g/L) sin(theta)."""

    def f(t, y):
        return np.array([y[1], -(GRAVEDAD / L) * np.sin(y[0])])

    return f


def energia_simple(Y, m, L):
    th, w = Y[:, 0], Y[:, 1]
    return 0.5 * m * (L * w) ** 2 + m * GRAVEDAD * L * (1 - np.cos(th))


def pendulo_doble(m1, m2, L1, L2):
    """Estado y = [theta1, omega1, theta2, omega2] (ángulos desde la vertical)."""

    def f(t, y):
        th1, w1, th2, w2 = y
        d = th1 - th2
        den = 2 * m1 + m2 - m2 * np.cos(2 * d)
        a1 = (
            -GRAVEDAD * (2 * m1 + m2) * np.sin(th1)
            - m2 * GRAVEDAD * np.sin(th1 - 2 * th2)
            - 2 * np.sin(d) * m2 * (w2**2 * L2 + w1**2 * L1 * np.cos(d))
        ) / (L1 * den)
        a2 = (
            2
            * np.sin(d)
            * (
                w1**2 * L1 * (m1 + m2)
                + GRAVEDAD * (m1 + m2) * np.cos(th1)
                + w2**2 * L2 * m2 * np.cos(d)
            )
        ) / (L2 * den)
        return np.array([w1, a1, w2, a2])

    return f


def energia_doble(Y, m1, m2, L1, L2):
    th1, w1, th2, w2 = Y.T
    d = th1 - th2
    T = 0.5 * m1 * (L1 * w1) ** 2 + 0.5 * m2 * (
        (L1 * w1) ** 2 + (L2 * w2) ** 2 + 2 * L1 * L2 * w1 * w2 * np.cos(d)
    )
    V = -(m1 + m2) * GRAVEDAD * L1 * np.cos(th1) - m2 * GRAVEDAD * L2 * np.cos(th2)
    return T + V


# ---------------------------------------------------------------------------
# 3. Newton-Raphson para sistemas  F(x) = 0
# ---------------------------------------------------------------------------
def newton_raphson(F, J, x0, tol=1e-8, max_iter=50):
    """Resuelve F(x) = 0 con Newton-Raphson.

    Devuelve (x, historial, convergio) donde historial es una lista de
    tuplas (iteración, x, ||F(x)||).
    """
    x = np.array(x0, dtype=float)
    res = np.linalg.norm(F(x))
    historial = [(0, x.copy(), res)]
    if res < tol:
        return x, historial, True
    for k in range(1, max_iter + 1):
        try:
            dx = np.linalg.solve(J(x), -F(x))
        except np.linalg.LinAlgError:  # Jacobiano singular
            return x, historial, False
        x = x + dx
        res = np.linalg.norm(F(x))
        historial.append((k, x.copy(), res))
        if not np.isfinite(res):
            return x, historial, False
        if res < tol:
            return x, historial, True
    return x, historial, False


# ---------------------------------------------------------------------------
# 4. Mecanismo de cuatro barras
#    r1 = bancada (O2-O4, sobre el eje x), r2 = manivela, r3 = acoplador,
#    r4 = balancín.  Incógnitas: theta3 (acoplador) y theta4 (balancín).
# ---------------------------------------------------------------------------
def lazo_cuatro_barras(theta2, r1, r2, r3, r4):
    """Devuelve F(x) y su Jacobiano para x = [theta3, theta4]."""

    def F(x):
        th3, th4 = x
        return np.array(
            [
                r2 * np.cos(theta2) + r3 * np.cos(th3) - r4 * np.cos(th4) - r1,
                r2 * np.sin(theta2) + r3 * np.sin(th3) - r4 * np.sin(th4),
            ]
        )

    def J(x):
        th3, th4 = x
        return np.array(
            [
                [-r3 * np.sin(th3), r4 * np.sin(th4)],
                [r3 * np.cos(th3), -r4 * np.cos(th4)],
            ]
        )

    return F, J


def cuatro_barras_analitica(theta2, r1, r2, r3, r4, signo=1):
    """Solución cerrada (intersección de dos circunferencias).

    Sirve como estimado inicial y como referencia para validar Newton-Raphson.
    Devuelve None si el mecanismo no se puede ensamblar con ese theta2.
    """
    A = np.array([r2 * np.cos(theta2), r2 * np.sin(theta2)])
    O4 = np.array([r1, 0.0])
    v = O4 - A
    d = float(np.hypot(v[0], v[1]))
    if d < 1e-12 or d > r3 + r4 or d < abs(r3 - r4):
        return None
    a = (r3**2 - r4**2 + d**2) / (2 * d)
    hh = np.sqrt(max(r3**2 - a**2, 0.0))
    M = A + a * v / d
    perp = np.array([-v[1], v[0]]) / d
    B = M + signo * hh * perp
    th3 = np.arctan2(B[1] - A[1], B[0] - A[0])
    th4 = np.arctan2(B[1], B[0] - r1)
    return np.array([th3, th4])


def barrido_cuatro_barras(thetas2, r1, r2, r3, r4, signo=1, tol=1e-10):
    """Resuelve el mecanismo para muchos theta2 usando continuación:
    la solución anterior es el estimado inicial del siguiente punto.

    Devuelve un arreglo (n, 2) con [theta3, theta4]; NaN donde no hay solución.
    """
    sol = np.full((len(thetas2), 2), np.nan)
    previo = None
    for i, t2 in enumerate(thetas2):
        if previo is None:
            previo = cuatro_barras_analitica(t2, r1, r2, r3, r4, signo)
            if previo is None:
                continue
        F, J = lazo_cuatro_barras(t2, r1, r2, r3, r4)
        x, _, ok = newton_raphson(F, J, previo, tol)
        if ok:
            sol[i] = x
            previo = x
        else:
            previo = None
    return sol
