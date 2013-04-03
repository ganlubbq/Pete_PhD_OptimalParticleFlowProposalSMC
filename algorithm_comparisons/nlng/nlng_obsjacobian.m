function [ H ] = nlng_obsjacobian( model, x )
%nlng_obsjacobian Calculate observation function jacobian for the
%nonlinear non-Gaussian benchmark model.

nl = model.alpha1 * model.alpha2 * x .* (x.^2).^(model.alpha2/2 - 1);
nl(isnan(nl)) = 0;
H = model.Hlin * diag(nl);

end

