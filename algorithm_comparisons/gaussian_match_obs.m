function [ y, H, R ] = gaussian_match_obs( x, p, Dp_p )
%gaussian_match_obs Select a Gaussian for the observation density with a
%particular value and gradient at point x.

% p is the density and Dp_p the ratio grad(density)/density

ds = length(x);

DTD = Dp_p'*Dp_p;

H = eye(ds);
sig = ds*numerical_lambertw(DTD*p^(-2/ds)/(2*pi*ds))/DTD;
R = sig*eye(ds);
y = x + sig*Dp_p;

end

