clc
clear all

syms alp x1 x2 x3

f1 = 3*x1 - cos(x2*x3) - 1/2;
f2 = x1^2 - 81*(x2 + 0.1)^2 + sin(x3) + 1.06;
f3 = exp(-x1*x2) + 20*x3 + (10*pi - 3) / 3;

Xi = [0, 0, 0]';

g = f1^2 + f2^2 + f3^2
XF_S = Steepest_Descent(g, Xi, 0.01, 200)


F = [f1, f2, f3];
XF_H = Homotopy(F, Xi, 200)
