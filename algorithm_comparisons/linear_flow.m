function [ A, b ] = linear_flow( lam, m, P, y, H, R )
%linear_flow Calculate the linear flow for a set of matrixes. Optimal for
%linear Gaussian models.

ds = size(m, 1);
I = eye(ds);

A = -0.5*P*H'*((R+lam*H*P*H')\H);
b = (I+2*lam*A)*((I+lam*A)*P*H'*(R\y)+A*m);

end
