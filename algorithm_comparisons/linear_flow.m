function [ AD, bD ] = linear_flow( lam, m, P, y, H, R, D )
%linear_flow Calculate the linear flow for a set of matrixes. Optimal for
%linear Gaussian models.

ds = size(m, 1);
I = eye(ds);

A = -0.5*P*H'*((R+lam*H*P*H')\H);
b = (I+2*lam*A)*((I+lam*A)*P*H'*(R\y)+A*m);

if ~all(D==0)
    AD = A - D*(inv(P)+lam*H'*(R\H));
    bD = b + D*(P\m + lam*H'*(R\y));
else
    AD = A;
    bD = b;
end

end
