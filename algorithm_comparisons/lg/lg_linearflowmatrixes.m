function [ x0_mat, m_mat, Hy_vec, x_cov, jac ] = lg_linearflowmatrixes( P, H, R, y, Dscale )
%LG_LINEARFLOWMATRIXES Summary of this function goes here
%   Detailed explanation goes here

% J is the jacobian of the transformation, |dx/dx0|

% Preliminaries
ds = size(P, 1);
I = eye(ds);

% Useful matrixes
St = I + P*H'*(R\H);
sqrtSt = sqrtm(St);
expdS = exp(-Dscale/2);

% H pseudo-inverse
[nr, nc] = size(H);
r = rank(H);
assert(r==min(nr,nc), 'H is not full rank');
Hpi = pinv(H);

% Find linear map
x0_mat = expdS*I/sqrtSt;
m_mat  = -x0_mat + inv(St);
Hy_vec = (I-inv(St))*Hpi*y;

if Dscale ~= 0
    % Covariance
    x_cov = (1-expdS^2)*(St\P);
    x_cov = (x_cov+x_cov')/2;
    x_cov = x_cov + 1E-8*eye(size(x_cov));
else
    x_cov = [];
end

% Calculate the Jacobian
jac = expdS/sqrt(det(St));

end

