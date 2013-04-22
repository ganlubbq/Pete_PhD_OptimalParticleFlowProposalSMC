% function [ y, R ] = gaussian_match_obs( x, Dp, D2p, H, R )
% %gaussian_match_obs Select a Gaussian for the observation density with a
% %particular gradient and Hessian of the log-density at point x.
% 
% % Check H dimension
% [m,n] = size(H);
% r = rank(H);
% assert(r==m, 'Haven''t solved/coded that case!')
% 
% % Pseudo-inverse
% Hpi = pinv(H);
% 
% % Covariance
% R = inv( -Hpi'*D2p*Hpi );
% 
% % Make it positive definite
% if ~isposdef(R)
%     R = R - (min(eig(R))-1E-4)*eye(size(R));
% end
% 
% assert(isposdef(R))
% 
% % Observation
% y = pinv(H'/R)*( Dp + H'*(R\H)*x );
% 
% end

function [ y, H, R ] = gaussian_match_obs( x, p, Dp_p, H, obs_mn )
%gaussian_match_obs Select a Gaussian for the observation density with a
%particular value and gradient at point x.

% p is the density and Dp_p the ratio grad(density)/density

% ds = length(x);
% 
% DTD = Dp_p'*Dp_p;
% 
% H = eye(ds);
% sig = ds*numerical_lambertw(DTD*p^(-2/ds)/(2*pi*ds))/DTD;
% R = sig*eye(ds);
% y = x + sig*Dp_p;

% H = eye(length(x));

[do,ds] = size(H);

Hpi = pinv(H);
DTD = sum((Dp_p'*Hpi).^2);

sig = ds*numerical_lambertw(DTD*p^(-2/ds)/(2*pi*ds))/DTD;
R = sig*eye(do);

y = (obs_mn-H*x) + sig*Hpi'*Dp_p;

end