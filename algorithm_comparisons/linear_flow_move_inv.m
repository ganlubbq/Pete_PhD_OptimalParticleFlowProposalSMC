function [ x, prob_ratio, drift, diffuse] = linear_flow_move_inv( t, t0, x0, m, invP, y, H, R, Dscale, zD )
%linear_flow_move Calculate the movement produced by a linear flow for a
%set of matrixes. Optimal for linear Gaussian models. Analytic integral of
%linear_flow.

% J is the jacobian of the transformation, |dx/dx0|

% Useful matrixes
HRH = H'*(R\H); HRH = (HRH+HRH')/2;

% OID approximations
[V,D] = eig(invP+t0*HRH);
D(D<0)=0;
D=pinv(D);
Sigma0 = V*D*V';

[V,D] = eig(invP+t*HRH);
D(D<0)=0;
D=pinv(D);
Sigmat = V*D*V';


% Sigma0 = pinv(invP+t0*HRH);
% Sigmat = pinv(invP+t*HRH);
Sigma0 = (Sigma0+Sigma0)/2;
Sigmat = (Sigmat+Sigmat)/2;

mu0 = Sigma0*(t0*H'*(R\y)+invP*m);
mut = Sigmat*(t*H'*(R\y)+invP*m);
sqrtSigmat = sqrtm(Sigmat);

% Rotation
% [rot, tmp] = qr(randn(ds));
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
