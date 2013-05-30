function [ drift ] = linear_drift( t, x, m, P, y, H, R, Dscale )
%linear_drift Calculate the drift for a linear flow for a set of matrixes.

% Preliminaries
ds = size(m, 1);
I = eye(ds);

% Useful matrixes
HRH = H'*(R\H); HRH = (HRH+HRH')/2;
St = I + t*P*HRH;

% Drift
mu = St\(m+t*P*H'*(R\y));
drift = 0.5*( ((St^2)\H')*(R\(y-H*m)) + St\H'*(R\(y-H*x)) - Dscale*(x-mu) );

end

