function xf = Newton_Unid(x0, f, epsilon)
    % f es una funcion simbolica de lambda
    syms alp
    p = [x0];
    k = 1;
    f_p = diff(f, alp);
    f_pp = diff(f_p, alp);
    p(k + 1) = p(k) - double(subs(f_p, alp, p(k))) / double(subs(f_pp, alp, p(k)));
    while(abs(p(k+1) - p(k)) > epsilon)
        k = k + 1;
        p(k + 1) = p(k) - double(subs(f_p, alp, p(k))) / double(subs(f_pp, alp, p(k)));
    end
    xf = p(k+1);
end