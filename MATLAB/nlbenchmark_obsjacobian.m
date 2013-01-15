function [ H ] = nlbenchmark_obsjacobian( model, x )
%NLBENCHMARK_OBSJACOBIAN Summary of this function goes here
%   Detailed explanation goes here

nl = model.alpha1 * model.alpha2 * x .* (x.^2).^(model.alpha2/2 - 1);
nl(isnan(nl)) = 0;
H = diag(nl);
% H = model.Hlin * nl;

end

