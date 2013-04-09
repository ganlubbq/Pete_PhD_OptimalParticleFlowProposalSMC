function [ y, H, R ] = gaussian_match_obs( x, Dp, D2p )
%gaussian_match_obs Select a Gaussian for the observation density with a
%particular gradient and Hessian of the log-density at point x.

ds = length(x);

% We get to choose H (must be nvertible), so how about I
H = eye(ds);

% Covariance
R = -inv(D2p);

% Make it positive definite
if ~isposdef(R)
    R = R - (min(eig(R))-1E-4)*eye(ds);
end

assert(isposdef(R))

% Mean
y = x - D2p\Dp; 

end

