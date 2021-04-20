###################################################################################
# Librerias Importadas ############################################################

import numpy as np
import matplotlib.pyplot as plt


###################################################################################
# Definicion de Funciones y Metodos ###############################################

# Solucion Analitica
def y(t, m, k, c):
    return m / (1 + m * c * np.exp(-k * m * t))


# Expresion para dy/dt = f(y, t)
def f(t, y, k, m):
    return k * (m - y) * y


# Metodo de Euler
def Euler(f, a, b, y0, n, k, m):
    t = np.zeros(n + 1)
    y = np.zeros(n + 1)
    t[0] = a
    y[0] = y0
    h = (b - a) / n
    for i in range(n):
        t[i + 1] = t[i] + h
        y[i + 1] = y[i] + h * f(t[i], y[i], k, m)
    return t, y 


# Metodo de Euler Mejorado
def Euler_M(f, a, b, y0, n, k, m):
    t = np.zeros(n + 1)
    y = np.zeros(n + 1)
    t[0] = a
    y[0] = y0
    h = (b - a) / n
    for i in range(n):
        fi = f(t[i], y[i], k, m)
        fih = f(t[i] + h, y[i] + h * fi, k, m)
        t[i + 1] = t[i] + h
        y[i + 1] = y[i] + (h / 2) * (fi + fih) 
    return t, y 


# Metodo del Punto Medio
def Midpoint(f, a, b, y0, n, k, m):
    t = np.zeros(n + 1)
    y = np.zeros(n + 1)
    t[0] = a
    y[0] = y0
    h = (b - a) / n
    for i in range(n):
        fi = f(t[i], y[i], k, m)
        t[i + 1] = t[i] + h
        y[i + 1] = y[i] + h * f(t[i] + (h / 2), y[i] + (h / 2) * fi, k, m)
    return t, y


# Metodo de Runge Kutta
def Runge_K(f, a, b, y0, n, k, m):
    t = np.zeros(n + 1)
    y = np.zeros(n + 1)
    t[0] = a
    y[0] = y0
    h = (b - a) / n
    for i in range(n):
        t[i + 1] = t[i] + h
        k1 = f(t[i], y[i], k, m)
        k2 = f(t[i] + (h / 2), y[i] + (h / 2) * k1, k, m)
        k3 = f(t[i] + (h / 2), y[i] + (h / 2) * k2, k, m)
        k4 = f(t[i] + h, y[i] + h * k3, k, m)
        y[i + 1] = y[i] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return t, y


###################################################################################
# Bloque Principal de Instrucciones ###############################################

# Parametros del Problema
m = 100000
k = 2e-6
y0 = 1000
a = 0
b = 30
n = 60

# Constante de Integracion
c = (m - y0) / (y0 * m)

# Ejecucion de los Metodos Numericos
tE, yE = Euler(f, a, b, y0, n, k, m)
tEM, yEM = Euler_M(f, a, b, y0, n, k, m)
tM, yM = Midpoint(f, a, b, y0, n, k, m)
tRK, yRK = Runge_K(f, a, b, y0, n, k, m)

# Solucion Analitica
yA = [y(i, m, k, c) for i in tE]



# GRAFICAS IMPORTANTES


# w(t) o y(t) vs t

plt.figure(1)
plt.plot(tE, yA, 'b')
plt.plot(tE, yE, 'r')
plt.plot(tEM, yEM, 'g')
plt.plot(tM, yM, 'c')
plt.plot(tRK, yRK, 'm')
plt.grid()
plt.xlabel("t", fontsize = 10)
plt.ylabel("y(t) o w(t)", fontsize = 10)
plt.title("Solutions y(t) and w(t) vs t, n = 60, h = 0.5", fontsize = 15)
plt.legend(['Analitic', 'Euler', 'Euler_M', 'Midpoint', 'Runge_K'])
# figManager = plt.get_current_fig_manager()
# figManager.window.showMaximized()
plt.pause(1)
plt.savefig("w(t)y(t)_vs_t.PNG")
plt.show()


# |w(t) - y(t)| vs t

plt.figure(2)
plt.plot(tE, np.log(np.absolute(yE - yA)), 'r')
plt.plot(tEM, np.log(np.absolute(yEM - yA)), 'g')
plt.plot(tM, np.log(np.absolute(yM - yA)), 'c')
plt.plot(tRK, np.log(np.absolute(yRK - yA)), 'm')
plt.grid()
plt.xlabel("t", fontsize = 10)
plt.ylabel("log|w(t) - y(t)|", fontsize = 10)
plt.title("Absolute Error log|w(t) - y(t)| vs t, n = 60, h = 0.5", fontsize = 15)
plt.legend(['Euler', 'Euler_M', 'Midpoint', 'Runge_K'])
# figManager = plt.get_current_fig_manager()
# figManager.window.showMaximized()
plt.pause(1)
plt.savefig("log|w(t)-y(t)|_vs_t.PNG")
plt.show()


# |w(t) - y(t)| / |y(t)| vs t

plt.figure(3)
plt.plot(tE, np.log(np.divide(np.absolute(yE - yA), np.absolute(yA))), 'r')
plt.plot(tEM, np.log(np.divide(np.absolute(yEM - yA), np.absolute(yA))), 'g')
plt.plot(tM, np.log(np.divide(np.absolute(yM - yA), np.absolute(yA))), 'c')
plt.plot(tRK, np.log(np.divide(np.absolute(yRK - yA), np.absolute(yA))), 'm')
plt.grid()
plt.xlabel("t", fontsize = 10)
plt.ylabel("log(|w(t) - y(t)| / |y(t)|)", fontsize = 10)
plt.title(" Relative Error log(|w(t) - y(t)| / |y(t)|) vs t, n = 60, h = 0.5", fontsize = 15)
plt.legend(['Euler', 'Euler_M', 'Midpoint', 'Runge_K'])
# figManager = plt.get_current_fig_manager()
# figManager.window.showMaximized()
plt.pause(1)
plt.savefig("log(|w(t)-y(t)|div|y(t)|)_vs_t.PNG")
plt.show()


# |w(t) - y(t)| vs h, t = 30, h \in [0.0005, 0.5], n \in [60, 60000]

t_fixed = 30
n_s = [60, 200, 400, 600, 2000, 4000, 6000, 20000, 40000, 60000]
h_s = [t_fixed / j for j in n_s]
absE = []
absEM = []
absM = []
absRK = []
for n_i in n_s:
    tE_ni, yE_ni = Euler(f, a, b, y0, n_i, k, m)
    tEM_ni, yEM_ni = Euler_M(f, a, b, y0, n_i, k, m)
    tM_ni, yM_ni = Midpoint(f, a, b, y0, n_i, k, m)
    tRK_ni, yRK_ni = Runge_K(f, a, b, y0, n_i, k, m)
    yA_ni = [y(i, m, k, c) for i in tE_ni]
    absE.append(abs(yE_ni[-1] - yA_ni[-1]))
    absEM.append(abs(yEM_ni[-1] - yA_ni[-1]))
    absM.append(abs(yM_ni[-1] - yA_ni[-1]))
    absRK.append(abs(yRK_ni[-1] - yA_ni[-1]))

# Bound Theorem 5.9
M = 384.9
L = 0.2
h = 0.5
cota = [(h_i * M) / (2 * L) * (np.exp(L * t_fixed) - 1) for h_i in h_s]

plt.figure(4)
plt.plot(h_s, np.log(absE), 'r')
plt.plot(h_s, np.log(absEM), 'g')
plt.plot(h_s, np.log(absM), 'c')
plt.plot(h_s, np.log(absRK), 'm')
plt.plot(h_s, np.log(cota), 'y')
plt.grid()
plt.xlabel("h", fontsize = 10)
plt.ylabel("log|w(t = 30) - y(t = 30)|", fontsize = 10)
plt.title("Absolute Error log|w(t) - y(t)| vs h, t = 30", fontsize = 15)
plt.legend(['Euler', 'Euler_M', 'Midpoint', 'Runge_K', 'Bound Error to Euler'])
# figManager = plt.get_current_fig_manager()
# figManager.window.showMaximized()
plt.pause(1)
plt.savefig("log|w(t)-y(t)|_vs_h.PNG")
plt.show()