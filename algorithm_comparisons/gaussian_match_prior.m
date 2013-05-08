function [ y, Q ] = gaussian_match_prior( x, p, Dp_p )
%gaussian_match_prior Select a Gaussian for the prior density with a
%particular value and gradient at point x.

% p is the density and Dp_p the ratio grad(density)/density

ds = length(x);

DTD = Dp_p'*Dp_p;

sig = ds*numerical_lambertw(DTD*p^(-2/ds)/(2*pi*ds))/DTD;
Q = sig*eye(ds);
y = x + sig*Dp_p;

end