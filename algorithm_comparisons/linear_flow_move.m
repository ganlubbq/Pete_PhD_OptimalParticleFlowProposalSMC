function [ x, J, prob_ratio] = linear_flow_move( t, t0, x0, m, P, y, H, R, Dscale )
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

% dS = St-S0;
% expdS = expm(-Dscale*dS/2);
% expdS = expm(-Dscale*(t-t0)*I/2);
expdS = exp(-Dscale*(t-t0)/2);

% H pseudo-inverse
assert(r==min(nr,nc), 'H is not full rank');
Hpi = pinv(H);

% Find x
F = expdS*(sqrtSt\sqrtS0);
r = Hpi*y - St\(Hpi*y-m) + expdS*(sqrtSt\(sqrtS0\( (Hpi*y-m) - S0*Hpi*y )));
x_mn = F*x0+r;
% x_mn = Hpi*y - St\Hpi*(y-H*m) + (expdS/sqrtSt)*(sqrtS0\( Hpi*(y-H*m) - S0*(Hpi*y-x0) ));
% x = Mt\( M0*x0 ...
%     - (sqrtSt\expSt - sqrtS0\expS0)*Hpi*(y-H*m) ...
%     + (sqrtSt*expSt - sqrtS0*expS0)*Hpi*y   );

if Dscale ~= 0
    % Stochastic bit
    x_vr = (1-expdS^2)*(St\P);
    x_vr = (x_vr+x_vr')/2;
    x_vr = x_vr + 1E-8*eye(size(x_vr));
    x = mvnrnd(x_mn', x_vr)';
%     Vr = (expm(Dscale*St) - expm(Dscale*S0))*P;
%     dIs = mvnrnd(zeros(size(x')), Vr)';
%     x = x + Mt\dIs;

else
    x = x_mn;
    
end

% Calculate the Jacobian
J = sqrt(det(S0)/det(St))*det(expdS);

if Dscale ~= 0
    % Probabilities
    C = x_vr;
    [art_mn, art_vr] = kf_update(m, P, x-r, F, C);
    art_prob = loggausspdf(x0, art_mn, art_vr);
    flow_prob = loggausspdf(x, x_mn, x_vr);
    prob_ratio = art_prob-flow_prob;
else
    prob_ratio = 0;
end

end
