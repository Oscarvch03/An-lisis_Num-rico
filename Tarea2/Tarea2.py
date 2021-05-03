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

# n = 500
# k = n - 1
# alpha = 1/2
# A = A_k(k, alpha)
# b = b_k(k, alpha)
# # print(A)
# # print(b)
# # print()
# p0 = 1
# pn = 0


## a)


# # 1)  linalg.inv

# x1, t1 = method_inv(A, b)
# # print(x1)
# # print("t1: ", t1)
# # print()

# # 2) linalg.lstsq

# x2, t2 = method_lstsq(A, b)
# # print(x2)
# # print("t2: ", t2)
# # print()

# # 3) linalg.lu

# x3, t3 = method_lu(A, b)
# # print(x3)
# # print("t3: ", t3)
# # print()

# # 4) linalg.cholesky (A def positive matrix)

# x4, t4 = method_cholesky(A, b)
# # print(x4)
# # print("t4: ", t4)
# # print()

# # 5) linalg.solve

# x5, t5 = method_solve(A, b)
# # print(x5)
# # print("t5: ", t5)
# # print()



## b) 


# 1) Pj vs j/n

# P1 = [p0] + list(x1) + [pn]
# P2 = [p0] + list(x2) + [pn]
# P3 = [p0] + list(x3) + [pn]
# P4 = [p0] + list(x4) + [pn]
# P5 = [p0] + list(x5) + [pn]
# j_n = [i / n for i in range(n + 1)]
# # print(P1)
# # print(P2)
# # print(P3)
# # print(P4)
# # print(P5)

# plt.figure(1)
# plt.plot(j_n, P1, 'b')
# plt.plot(j_n, P2, 'r')
# plt.plot(j_n, P3, 'g')
# plt.plot(j_n, P4, 'c')
# plt.plot(j_n, P5, 'm')
# plt.grid()
# plt.xlabel("j / n", fontsize = 10)
# plt.ylabel("Pj", fontsize = 10)
# plt.title("Pj vs j/n, n = 500")
# plt.legend(['linalg.inv', 'linalg.lstsq', 'linalg.lu', 'linalg.cholesky', 'linalg.solve'])
# plt.pause(1)
# plt.savefig("Pj_vs_jdivn_all_methods.PNG")
# plt.show()


# 2) Wall-time vs n (Average with 10 rounds)

# ns = [10, 20, 50, 100, 200, 500, 1000]
# ks = [i - 1 for i in ns]
# w1 = np.zeros(len(ks))
# w2 = np.zeros(len(ks))
# w3 = np.zeros(len(ks))
# w4 = np.zeros(len(ks))
# w5 = np.zeros(len(ks))
# for i in range(10):
#     walltime1 = []
#     walltime2 = []
#     walltime3 = []
#     walltime4 = []
#     walltime5 = []
#     for m in ks:
#         Am = A_k(m, alpha)
#         bm = b_k(m, alpha)
#         x1m, t1m = method_inv(Am, bm)
#         x2m, t2m = method_lstsq(Am, bm)
#         x3m, t3m = method_lu(Am, bm)
#         x4m, t4m = method_cholesky(Am, bm)
#         x5m, t5m = method_solve(Am, bm)
#         walltime1.append(t1m)
#         walltime2.append(t2m)
#         walltime3.append(t3m)
#         walltime4.append(t4m)
#         walltime5.append(t5m)
#     w1 += walltime1
#     w2 += walltime2
#     w3 += walltime3
#     w4 += walltime4
#     w5 += walltime5

# w1 /= 10
# w2 /= 10
# w3 /= 10
# w4 /= 10
# w5 /= 10

# plt.figure(2)
# plt.plot(ks, w1, 'b')
# plt.plot(ks, w2, 'r')
# plt.plot(ks, w3, 'g')
# plt.plot(ks, w4, 'c')
# plt.plot(ks, w5, 'm')
# plt.grid()
# plt.xlabel("n", fontsize = 10)
# plt.ylabel("Wall Time", fontsize = 10)
# plt.title("Wall Time vs n, n = 10, 20, 50, 100, 200, 500, 1000")
# plt.legend(['linalg.inv', 'linalg.lstsq', 'linalg.lu', 'linalg.cholesky', 'linalg.solve'])
# plt.pause(1)
# plt.savefig("WallTime_vs_n_all_methods.PNG")
# plt.show()



## d) Fastest methods (linalg.inv and linalg.solve)


# 1) Pj vs j/n

# n = 500
# k = n - 1
# p0 = 1
# pn = 0

# alpha1 = 0
# A1 = A_k(k, alpha1)
# b1 = b_k(k, alpha1)
# x1, t1 = method_inv(A1, b1)
# x2, t2 = method_solve(A1, b1)
# P1 = [p0] + list(x1) + [pn]
# P2 = [p0] + list(x2) + [pn]
# j_n = [i / n for i in range(n + 1)]
# plt.figure(3)
# plt.plot(j_n, P1, 'b')
# plt.plot(j_n, P2, 'm')
# plt.grid()
# plt.xlabel("j / n", fontsize = 10)
# plt.ylabel("Pj", fontsize = 10)
# plt.title("Pj vs j/n, n = 500, alpha = 0")
# plt.legend(['linalg.inv', 'linalg.solve'])
# plt.pause(1)
# plt.savefig("Pj_vs_jdivn_best_methods_alpha1.PNG")
# plt.show()

# alpha2 = 1/3
# A2 = A_k(k, alpha2)
# b2 = b_k(k, alpha2)
# x1, t1 = method_inv(A2, b2)
# x2, t2 = method_solve(A2, b2)
# P1 = [p0] + list(x1) + [pn]
# P2 = [p0] + list(x2) + [pn]
# j_n = [i / n for i in range(n + 1)]
# plt.figure(3)
# plt.plot(j_n, P1, 'b')
# plt.plot(j_n, P2, 'm')
# plt.grid()
# plt.xlabel("j / n", fontsize = 10)
# plt.ylabel("Pj", fontsize = 10)
# plt.title("Pj vs j/n, n = 500, alpha = 1/3")
# plt.legend(['linalg.inv', 'linalg.solve'])
# plt.pause(1)
# plt.savefig("Pj_vs_jdivn_best_methods_alpha2.PNG")
# plt.show()

# alpha3 = 2/3
# A3 = A_k(k, alpha3)
# b3 = b_k(k, alpha3)
# x1, t1 = method_inv(A3, b3)
# x2, t2 = method_solve(A3, b3)
# P1 = [p0] + list(x1) + [pn]
# P2 = [p0] + list(x2) + [pn]
# j_n = [i / n for i in range(n + 1)]
# plt.figure(3)
# plt.plot(j_n, P1, 'b')
# plt.plot(j_n, P2, 'm')
# plt.grid()
# plt.xlabel("j / n", fontsize = 10)
# plt.ylabel("Pj", fontsize = 10)
# plt.title("Pj vs j/n, n = 500, alpha = 2/3")
# plt.legend(['linalg.inv', 'linalg.solve'])
# plt.pause(1)
# plt.savefig("Pj_vs_jdivn_best_methods_alpha3.PNG")
# plt.show()

# alpha4 = 1
# A4 = A_k(k, alpha4)
# b4 = b_k(k, alpha4)
# x1, t1 = method_inv(A4, b4)
# x2, t2 = method_solve(A4, b4)
# P1 = [p0] + list(x1) + [pn]
# P2 = [p0] + list(x2) + [pn]
# j_n = [i / n for i in range(n + 1)]
# plt.figure(3)
# plt.plot(j_n, P1, 'b')
# plt.plot(j_n, P2, 'm')
# plt.grid()
# plt.xlabel("j / n", fontsize = 10)
# plt.ylabel("Pj", fontsize = 10)
# plt.title("Pj vs j/n, n = 500, alpha = 1")
# plt.legend(['linalg.inv', 'linalg.solve'])
# plt.pause(1)
# plt.savefig("Pj_vs_jdivn_best_methods_alpha4.PNG")
# plt.show()


# 2) Pj vs alpha

n = 500
k = n - 1
p0 = 1
pn = 0
j = [1, n / 2, n - 1]
alphas = np.linspace(0, 1, num = 9)
# print(alphas)
plt.figure(4)
for a in alphas:
    Aa = A_k(k, a)
    ba = b_k(k, a)
    x1, t1 = method_inv(Aa, ba)
    x2, t2 = method_solve(Aa, ba)
    P1 = [p0] + list(x1) + [pn]
    P2 = [p0] + list(x2) + [pn]
    P1_j = [P1[int(i)] for i in j]
    P2_j = [P2[int(i)] for i in j]
    plt.plot([a for i in range(3)], P1_j, "bo")
    plt.plot([a for i in range(3)], P2_j, "mo")
plt.xlabel("alpha", fontsize = 10)
plt.ylabel("Pj", fontsize = 10)
plt.title("Pj vs alpha, n = 500; j = 1, n/2, n-1")
plt.legend(['linalg.inv', 'linalg.solve'])
plt.pause(1)
plt.savefig("Pj_vs_alpha_best_methods.PNG")
plt.show()