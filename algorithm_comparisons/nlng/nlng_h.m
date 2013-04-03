function [ y_mn ] = nlng_h( model, x )
%nlng_h Deterministic observation function for the nonlinear non-Gaussian
%benchmark model.

nl_mn = model.alpha1 * abs(x).^model.alpha2;
y_mn = model.Hlin * nl_mn;

end
