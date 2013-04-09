function [ m, P ] = gaussian_match_prior( x, Dp, D2p )
%gaussian_match_prior Select a Gaussian for the prior density with a
%particular gradient and Hessian of the log-density at point x.

% Covariance
P = -inv(D2p);

assert(isposdef(P));

% Mean
m = x - D2p\Dp; 

end