function [ x, mut, Sigmat, prob_ratio, drift, diffuse] = modified_linear_flow_move( t, t0, x0, mu0, Sigma0, ymhx, H, R, Dscale, zD )
%linear_flow_move Calculate the movement produced by a linear flow for a
%set of matrixes. Optimal for linear Gaussian models. Analytic integral of
%linear_flow.

% J is the jacobian of the transformation, |dx/dx0|

% Update
S = H*Sigma0*H' + R/(t-t0);
C = Sigma0*H';
K = C/S;

mut = mu0 + K*(ymhx - H*(mu0-x0));
Sigmat = Sigma0 - K*S*K';
sqrtSigmat = sqrtm(Sigmat);


% Transformation
Gam = exp(-0.5*Dscale*(t-t0))*sqrtSigmat/sqrtm(Sigma0);
x_mn = mut + Gam*(x0-mu0);

% Randomness
if Dscale > 0
    sqrtOme = sqrt(1-exp(-Dscale*(t-t0)))*sqrtSigmat;
    x = x_mn + sqrtOme*zD;
    prob_ratio = exp(loggausspdf(x0,mu0,Sigma0)-loggausspdf(x,mut,Sigmat));
else
    x = x_mn;
    prob_ratio = 1/sqrt( det(eye(size(Sigma0)) + (t-t0)*Sigma0*H'*(R\H)) );
end

% Drift at and diffusion end state
drift = Sigmat*(H'/R)*( (ymhx-H*(mut-x0))-0.5*H*(x-mut) );
diffuse = 0;

end
