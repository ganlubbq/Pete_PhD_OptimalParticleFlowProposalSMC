function [ drift ] = linear_drift( t, x, m, P, y, H, R, Dscale )
%linear_drift Calculate the drift for a linear flow for a set of matrixes.

% Useful matrixes
HRH = H'*(R\H); HRH = (HRH+HRH')/2;

% OID approx
Sigmat = inv(inv(P)+t*HRH);
mut = Sigmat*(t*H'*(R\y)+P\m);

% Drift
drift =  Sigmat*(H'/R)*( (y-H*mut)-0.5*H*(x-mut) ) - Dscale*(x-mut);

end

