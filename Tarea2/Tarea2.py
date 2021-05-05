###################################################################################
# Librerias Importadas ############################################################

import numpy as np
import scipy.linalg as scl
import time as tm
import matplotlib.pyplot as plt


###################################################################################
# Definicion de Funciones y Metodos ###############################################

def A_k(k, alpha):
    m = np.identity(k)
    for i in range(0, k - 1):
        m[i, i + 1] = alpha - 1
        m[i + 1, i] = -alpha
    return m


def b_k(k, alpha):
    v = np.zeros(k)
    v[0] = alpha
    return v


def method_inv(A, b):
    t11 = tm.time()
    A_inv = scl.inv(A)
    x1 = np.dot(A_inv, b)
    t12 = tm.time() - t11
    return x1, t12


def method_lstsq(A, b):
    t21 = tm.time()
    x2, res, rank, s = scl.lstsq(A, b)
    t22 = tm.time() - t21
    return x2, t22


def method_lu(A, b):
    t31 = tm.time()
    p, l, u = scl.lu(A)
    l_inv = scl.inv(l)
    u_inv = scl.inv(u)
    x3 = np.dot(u_inv, np.dot(l_inv, np.dot(p, b)))
    t32 = tm.time() - t31
    return x3, t32


def method_cholesky(A, b):
    t41 = tm.time() 
    L = scl.cholesky(A, lower = True)
    Lt = np.transpose(L)
    L_inv = scl.inv(L)
    Lt_inv = scl.inv(Lt)
    x4 = np.dot(Lt_inv, np.dot(L_inv, b))
    t42 = tm.time() - t41
    return x4, t42


def method_solve(A, b):
    t51 = tm.time()
    x5 = scl.solve(A, b)
    t52 = tm.time() - t51
    return x5, t52


###################################################################################
# Bloque Principal de Instrucciones ###############################################

n = 500
k = n - 1
alpha = 1/2
A = A_k(k, alpha)
b = b_k(k, alpha)
# print(A)
# print(b)
# print()
p0 = 1
pn = 0


## a)


# 1)  linalg.inv

x1, t1 = method_inv(A, b)
# print(x1)
# print("t1: ", t1)
# print()

# 2) linalg.lstsq

x2, t2 = method_lstsq(A, b)
# print(x2)
# print("t2: ", t2)
# print()

# 3) linalg.lu

x3, t3 = method_lu(A, b)
# print(x3)
# print("t3: ", t3)
# print()

# 4) linalg.cholesky (A def positive matrix)

x4, t4 = method_cholesky(A, b)
# print(x4)
# print("t4: ", t4)
# print()

# 5) linalg.solve

x5, t5 = method_solve(A, b)
# print(x5)
# print("t5: ", t5)
# print()


# Orden del Error más grande
max_x1_x2 = max(np.abs(x1 - x2))
max_x1_x3 = max(np.abs(x1 - x3))
max_x1_x4 = max(np.abs(x1 - x4))
max_x1_x5 = max(np.abs(x1 - x5))
max_x2_x3 = max(np.abs(x2 - x3))
max_x2_x4 = max(np.abs(x2 - x4))
max_x2_x5 = max(np.abs(x2 - x5))
max_x3_x4 = max(np.abs(x3 - x4))
max_x3_x5 = max(np.abs(x3 - x5))
max_x4_x5 = max(np.abs(x4 - x5))
maxs = [max_x1_x2, max_x1_x3, max_x1_x4, max_x1_x5, max_x2_x3, max_x2_x4, max_x2_x5, max_x3_x4, max_x3_x5, max_x4_x5]
for i in maxs:
    print(i)
max_def = max(maxs)
max_idx = maxs.index(max_def)
print("Max Error: ", max_def)
print("Max Idx: ", max_idx)
print("Between Method 2: lstsq and Method 4: Cholesky")


## b) 


# 1) Pj vs j/n

P1 = [p0] + list(x1) + [pn]
P2 = [p0] + list(x2) + [pn]
P3 = [p0] + list(x3) + [pn]
P4 = [p0] + list(x4) + [pn]
P5 = [p0] + list(x5) + [pn]
j_n = [i / n for i in range(n + 1)]
# print(P1)
# print(P2)
# print(P3)
# print(P4)
# print(P5)

plt.figure(1)
plt.plot(np.linspace(0, 1, num = 6), [P1[i] for i in np.linspace(0, 500, num = 6).astype(int)], 'bo', ms = 7)
plt.plot(np.linspace(0, 1, num = 6) + 0.04, [P2[i] if i < 500 else np.nan for i in (np.linspace(0, 500, num = 6).astype(int) + 20)] , 'rv', ms = 7)
plt.plot(np.linspace(0, 1, num = 6) + 0.08, [P3[i] if i < 500 else np.nan for i in (np.linspace(0, 500, num = 6).astype(int) + 40)] , 'g^', ms = 7)
plt.plot(np.linspace(0, 1, num = 6) + 0.12, [P4[i] if i < 500 else np.nan for i in (np.linspace(0, 500, num = 6).astype(int) + 60)] , 'cs', ms = 7)
plt.plot(np.linspace(0, 1, num = 6) + 0.16, [P5[i] if i < 500 else np.nan for i in (np.linspace(0, 500, num = 6).astype(int) + 80)] , 'm*', ms = 7)
plt.grid()
plt.xlabel("j / n", fontsize = 10)
plt.ylabel("Pj", fontsize = 10)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title("Pj vs j/n, n = 500")
plt.legend(['linalg.inv', 'linalg.lstsq', 'linalg.lu', 'linalg.cholesky', 'linalg.solve'])
plt.pause(1)
plt.savefig("Pj_vs_jdivn_all_methods.PNG")
plt.show()


# 2) Wall-time vs n (Average with 10 rounds)

alpha = 1/2
ns = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
ks = [i - 1 for i in ns]
w1 = np.zeros(len(ks))
w2 = np.zeros(len(ks))
w3 = np.zeros(len(ks))
w4 = np.zeros(len(ks))
w5 = np.zeros(len(ks))
rounds = 10
for i in range(rounds):
    walltime1 = []
    walltime2 = []
    walltime3 = []
    walltime4 = []
    walltime5 = []
    for m in ks:
        Am = A_k(m, alpha)
        bm = b_k(m, alpha)
        x1m, t1m = method_inv(Am, bm)
        x2m, t2m = method_lstsq(Am, bm)
        x3m, t3m = method_lu(Am, bm)
        x4m, t4m = method_cholesky(Am, bm)
        x5m, t5m = method_solve(Am, bm)
        walltime1.append(t1m)
        walltime2.append(t2m)
        walltime3.append(t3m)
        walltime4.append(t4m)
        walltime5.append(t5m)
        print("It: ", i, m)
    w1 += walltime1
    w2 += walltime2
    w3 += walltime3
    w4 += walltime4
    w5 += walltime5

w1 /= rounds
w2 /= rounds
w3 /= rounds
w4 /= rounds
w5 /= rounds

plt.figure(2)
plt.plot(np.log(ks), np.log(w1), 'b')
plt.plot(np.log(ks), np.log(w2), 'r')
plt.plot(np.log(ks), np.log(w3), 'g')
plt.plot(np.log(ks), np.log(w4), 'c')
plt.plot(np.log(ks), np.log(w5), 'm')
plt.grid()
plt.xlabel("log(n)", fontsize = 10)
plt.ylabel("log(Wall Time)", fontsize = 10)
plt.title("log(Wall Time) vs log(n), \n n = 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000")
plt.legend(['linalg.inv', 'linalg.lstsq', 'linalg.lu', 'linalg.cholesky', 'linalg.solve'])
plt.pause(1)
plt.savefig("WallTime_vs_n_all_methods.PNG")
plt.show()



## d) Fastest methods (linalg.inv and linalg.solve)


1) Pj vs j/n

n = 500
k = n - 1
p0 = 1
pn = 0
j_n = [i / n for i in range(n + 1)]
# alphas = np.linspace(0, 1, num = 9)
alphas = [0, 1/3, 0.49, 0.497, 0.499, 0.5, 0.501, 0.503, 0.51, 2/3, 1]
colors = ["b", "g", "r", "c", "m", "y", "tab:purple", "tab:orange", "tab:pink", 'k', "tab:olive"]
errors = []
for j in range(len(alphas)):
    Aa = A_k(k, alphas[j])
    ba = b_k(k, alphas[j])
    x1, t1 = method_inv(Aa, ba)
    x2, t2 = method_solve(Aa, ba)
    P1 = [p0] + list(x1) + [pn]
    P2 = [p0] + list(x2) + [pn]
    errors.append(max(np.abs(x1 - x2)))
    plt.figure(3)
    plt.plot(j_n, P1, colors[j])
    plt.figure(4)
    plt.plot(j_n, P2, colors[len(colors) - j - 1])

print("Error entre inv y solve: ", max(errors))

plt.figure(3)
plt.grid()
plt.xlabel("j / n", fontsize = 10)
plt.ylabel("Pj", fontsize = 10)
plt.title("Pj vs j/n, n = 500; Method: linalg.inv")
plt.legend(["alpha = " + str(round(i, 3)) for i in alphas])
plt.pause(1)
plt.savefig("Pj_vs_jdivn_alphas_inv.PNG")

plt.figure(4)
plt.grid()
plt.xlabel("j / n", fontsize = 10)
plt.ylabel("Pj", fontsize = 10)
plt.title("Pj vs j/n, n = 500; Method: linalg.solve")
plt.legend(["alpha = " + str(round(i, 3)) for i in alphas])
plt.pause(1)
plt.savefig("Pj_vs_jdivn_alphas_solve.PNG")

plt.show()


# 2) Pj vs alpha

n = 500
k = n - 1
p0 = 1
pn = 0
js = [1, int(n / 2), n - 1]
js_inv = [[], [], []]
js_solv = [[], [], []]
alphas = np.linspace(0, 1, num = 9)
errors = []
# alphas = [0, 1/3, 0.49, 0.497, 0.499, 0.5, 0.501, 0.503, 0.51, 2/3, 1]
for j in range(len(alphas)):
    Aa = A_k(k, alphas[j])
    ba = b_k(k, alphas[j])
    x1, t1 = method_inv(Aa, ba)
    x2, t2 = method_solve(Aa, ba)
    P1 = [p0] + list(x1) + [pn]
    P2 = [p0] + list(x2) + [pn]
    errors.append(max(np.abs(x1 - x2)))
    for i in range(len(js)):
        js_inv[i].append(P1[js[i]])
        js_solv[i].append(P2[js[i]])

# print(js_inv)
# print(js_solv)
print("Error entre inv y solve: ", max(errors))

plt.figure(5)
plt.plot(alphas, js_inv[0], "ys--", ms = 10)
plt.plot(alphas, js_inv[1], "go--", ms = 8)
plt.plot(alphas, js_inv[2], "r*--", ms = 6)
plt.grid()
plt.xlabel("alpha", fontsize = 10)
plt.ylabel("Pj", fontsize = 10)
plt.title("Pj vs alpha, n = 500; j = 1, n/2, n-1; Method: linalg.inv")
plt.legend(["j = " + str(i) for i in js])
plt.pause(1)
plt.savefig("Pj_vs_alpha_js_inv.PNG")

plt.figure(6)
plt.plot(alphas, js_solv[0], "ms--", ms = 10)
plt.plot(alphas, js_solv[1], "co--", ms = 8)
plt.plot(alphas, js_solv[2], "k*--", ms = 6)
plt.grid()
plt.xlabel("alpha", fontsize = 10)
plt.ylabel("Pj", fontsize = 10)
plt.title("Pj vs alpha, n = 500; j = 1, n/2, n-1; Method: linalg.solve")
plt.legend(["j = " + str(i) for i in js])
plt.pause(1)
plt.savefig("Pj_vs_alpha_js_solve.PNG")

plt.show()
