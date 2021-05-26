function Xf = Descenso_Gradiente(f, X, epsilon, it)
    % Donde f es una funcion simbolica
    syms lmd x1 x2 x3 x4 x5 x6
    k = 0;
    x_k = X';
    while(double(norm(subs(gradient(f), [x1, x2, x3, x4, x5, x6], [x_k(1), x_k(2), x_k(3), x_k(4), x_k(5), x_k(6)]))) > epsilon)
        double(norm(subs(gradient(f), [x1, x2, x3, x4, x5, x6], [x_k(1), x_k(2), x_k(3), x_k(4), x_k(5), x_k(6)])))
        if(k == it)
            break
        end
        k
        d_k = -double(subs(gradient(f), [x1, x2, x3, x4, x5, x6], [x_k(1), x_k(2), x_k(3), x_k(4), x_k(5), x_k(6)]))
        f_aux = x_k + lmd * d_k;
        f_new = simplify(subs(f, [x1, x2, x3, x4, x5, x6], [f_aux(1), f_aux(2), f_aux(3), f_aux(4), f_aux(5), f_aux(6)]));
        lmd_k = Newton_Unid(0, f_new, 0.0001)
        x_k = double(x_k + lmd_k * d_k)
        rest1 = sum(x_k)
        k = k + 1;
        disp("-----------------------------------------------------------------")
    end
    k
    Xf = double(x_k);
    
end