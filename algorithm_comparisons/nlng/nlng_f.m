function [ new_x_mn ] = nlng_f( model, kk, x )
%nlng_f Deterministic transition function for the nonlinear non-Gaussian
%benchmark model.

new_x_mn = model.beta1 * x ...
         + model.beta2 * (sum(x)/(1+sum(x)^2)) ...
         + model.beta3 * cos(1.2*kk);

end

