function [ drift, diffuse ] = modified_linear_drift( t, x, mu, Sigma, ymhx, H, R, Dscale )
%linear_drift Calculate the drift for a linear flow for a set of matrixes.

% Drift at and diffusion end state
drift = Sigma*(H'/R)*( (ymhx-H*(mu-x))-0.5*H*(x-mu) );
diffuse = 0;

end

