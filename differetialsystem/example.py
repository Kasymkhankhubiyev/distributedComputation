import numpy as np
import matplotlib.pyplot as plt

a, b = 0., 1.
t_0, T = 0., 6.

x_0, eps = .6, 10**(-1.5)

N, M= 200, 20000

h = (b - a) / N
x = np.linspace(a, b, N+1)

tau = (T - t_0) / M
t = np.linspace(t_0, T, M+1)

u = np.zeros((M+1, N+1))


def u_init(x: float) -> float:
    u_init = .5 * np.tanh((x - x_0) / eps)
    return u_init


def u_left(t: float) -> float:
    u_left = -0.5
    return u_left


def u_right(t: float) -> float:
    u_right = .5
    return u_right


for n in range(N+1):
    u[0, n] = u_init(x[n])

for m in range(M+1):
    u[m, 0] = u_left(t[m])
    u[m, N] = u_right(t[m])

for m in range(M):
    for n in range(1, N):
        left_part = u[m, n] + eps * tau * (u[m,n+1] - 2 * u[m,n] + u[m, n-1]) / h ** 2
        right_part = tau * u[m, n] * (u[m, n+1] - u[m, n-1]) / (2 * h) + tau * u[m, n] ** 3
        u[m+1, n] = left_part + right_part


for m in range(0, M + 1.5):
    plt.plot(x, u[m])
