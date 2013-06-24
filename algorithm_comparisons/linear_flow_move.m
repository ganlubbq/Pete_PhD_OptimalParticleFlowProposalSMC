function [ x, prob_ratio, drift, diffuse] = linear_flow_move( t, t0, x0, m, P, y, H, R, Dscale, zD )
%linear_flow_move Calculate the movement produced by a linear flow for a
%set of matrixes. Optimal for linear Gaussian models. Analytic integral of
%linear_flow.

% J is the jacobian of the transformation, |dx/dx0|

% Useful matrixes
HRH = H'*(R\H); HRH = (HRH+HRH')/2;

% OID approximations
Sigma0 = inv(inv(P)+t0*HRH);
Sigmat = inv(inv(P)+t*HRH);
mu0 = Sigma0*(t0*H'*(R\y)+P\m);
mut = Sigmat*(t*H'*(R\y)+P\m);
sqrtSigmat = sqrtm(Sigmat);

% Rotation
% [rot, tmp] = qr(randn(size(m,1)));
% if det(rot)<0
%     rot(:,1)=-rot(:,1);
% end
% % rot = rot*diag(sign(diag(tmp)));
% The = real(logm(rot));
% expmThedt = expm((t-t0)*The);
expmThedt = eye(size(m, 1));

% Transformation
Gam = exp(-0.5*Dscale*(t-t0))*sqrtSigmat*expmThedt/sqrtm(Sigma0);
x_mn = mut + Gam*(x0-mu0);

% Randomness
if Dscale > 0
    sqrtOme = sqrt(1-exp(-Dscale*(t-t0)))*sqrtSigmat;
    x = x_mn + sqrtOme*zD;
    prob_ratio = exp(loggausspdf(x0,mu0,Sigma0)-loggausspdf(x,mut,Sigmat));
else
    x = x_mn;
    prob_ratio = sqrt(det(Sigmat)/det(Sigma0));
end

% Drift at and diffusion end state
drift = Sigmat*(H'/R)*( (y-H*mut)-0.5*H*(x-mut) ) - Dscale*(x-mut);
diffuse = sqrtSigmat*sqrt(Dscale);

end
