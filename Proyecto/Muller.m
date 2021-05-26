function xf = Muller(P, f, epsilon, N)
    syms alp
    p0 = P(1);
    p1 = P(2);
    p2 = P(3);
    h1 = p1 - p0;
    h2 = p2 - p1;
    d1 = double(subs(f, alp, p1) - subs(f, alp, p0)) / h1;
    d2 = double(subs(f, alp, p2) - subs(f, alp, p1)) / h2;
    d = (d2 - d1) / (h2 + h1);
    i = 3;
    pf = 0;
    while(i < N)
        b = d2 + h2 * d;
        D = sqrt(b ^ 2 - 4 * double(subs(f, alp, p2)) * d);
        
        if(abs(b - D) < abs(b + D))
            E = b + D;
        else
            E = b - D;
        end
        
        h = -2 * double(subs(f, alp, p2)) / E;
        p = p2 + h;
        
        if(abs(h) < epsilon)
           pf = p;
           break
        end
        p0 = p1;
        p1 = p2;
        p2 = p;
        h1 = p1 - p0;
        h2 = p2 - p1;
        d1 = double(subs(f, alp, p1) - subs(f, alp, p0)) / h1;
        d2 = double(subs(f, alp, p2) - subs(f, alp, p1)) / h2;
        d = (d2 - d1) / (h2 + h1);
        i = i + 1;
    end
    if(i >= N)
        xf = N;
    else
        xf = pf;
    end
end