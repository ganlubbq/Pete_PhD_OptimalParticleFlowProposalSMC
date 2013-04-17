function [ x, J, Z] = linear_flow_move( t, t0, x0, m, P, y, H, R, Dscale )
%linear_flow_move Calculate the movement produced by a linear flow for a
%set of matrixes. Optimal for linear Gaussian models. Analytic integral of
%linear_flow.

% J is the jacobian of the transformation, |dx/dx0|

% Preliminaries
ds = size(m, 1);
I = eye(ds);
[nr, nc] = size(H);
r = rank(H);

% Useful matrixes
S0 = I + t0*P*H'*(R\H);
St = I + t*P*H'*(R\H);

sqrtSt = sqrtm(St);
sqrtS0 = sqrtm(S0);

dS = St-S0;
expdS = expm(-Dscale*dS/2);

% H pseudo-inverse
assert(r==min(nr,nc), 'H is not full rank');
Hpi = pinv(H);

% Find x
F = (expdS/sqrtSt)*sqrtS0;
r = Hpi*y - St\Hpi*(y-H*m) + (expdS/sqrtSt)*(sqrtS0\( Hpi*(y-H*m) - S0*Hpi*y ));
x_mn = F*x0+r;
% x_mn = Hpi*y - St\Hpi*(y-H*m) + (expdS/sqrtSt)*(sqrtS0\( Hpi*(y-H*m) - S0*(Hpi*y-x0) ));
% x = Mt\( M0*x0 ...
%     - (sqrtSt\expSt - sqrtS0\expS0)*Hpi*(y-H*m) ...
%     + (sqrtSt*expSt - sqrtS0*expS0)*Hpi*y   );

if Dscale ~= 0
    % Stochastic bit
    Vr = (I-expdS^2)*(St\P);
    Vr = (Vr+Vr')/2;
    Vr = Vr + 1E-8*eye(size(Vr));
    x = mvnrnd(x_mn', Vr)';
%     Vr = (expm(Dscale*St) - expm(Dscale*S0))*P;
%     dIs = mvnrnd(zeros(size(x')), Vr)';
%     x = x + Mt\dIs;

else
    x = x_mn;
    
end

% Calculate the Jacobian
J = sqrt(det(S0)/det(St))*det(expdS);

if Dscale ~= 0
    % Artificial distribution
    Sigma = St\P;
    mu = Sigma*(P\m + t*H'*(R\y));
    C = Vr;
    Z = loggausspdf(x, F*mu+r, F*Sigma*F'+C);
else
    Z = 0;
end

end
