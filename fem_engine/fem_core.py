import numpy as np

_GAUSS_TABLE = {
    1: (np.array([0.0]), np.array([2.0])),
    2: (np.array([0.5773502691896257, -0.5773502691896257]), np.array([1.0, 1.0])),
    3: (np.array([0.7745966692414834, 0.0, -0.7745966692414834]), np.array([0.5555555555555556, 0.8888888888888888, 0.5555555555555556])),
    4: (np.array([0.8611363115940526, 0.3399810435848563, -0.3399810435848563, -0.8611363115940526]), np.array([0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538]))
}

def _generate_gauss_legendre_quadrature(n):
    def legendre(n, x):
        if n == 0: return 1.0
        elif n == 1: return x
        else: return ((2.0 * n - 1.0) * x * legendre(n - 1, x) - (n - 1) * legendre(n - 2, x)) / n

    def dlegendre(n, x):
        return (n / (x ** 2 - 1.0)) * ((x * legendre(n, x)) - legendre(n - 1, x))

    points = np.zeros(n)
    for i in range(1, n + 1):
        x = np.cos(np.pi * (i - 0.25) / (n + 0.5))
        error = 1e-15
        delta = 1.0
        while delta > error:
            dx = -legendre(n, x) / dlegendre(n, x)
            x = x + dx
            delta = abs(dx)
        points[i - 1] = x

    weights = np.zeros(n)
    for i in range(n):
        weights[i] = 2.0 / ((1.0 - points[i] ** 2) * dlegendre(n, points[i]) ** 2)
    return points, weights

def gauss_legendre_quadrature(n):
    if n in _GAUSS_TABLE:
        return _GAUSS_TABLE[n]

    return _generate_gauss_legendre_quadrature(n)