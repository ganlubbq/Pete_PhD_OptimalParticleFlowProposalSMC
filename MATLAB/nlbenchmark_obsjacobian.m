function [ H ] = nlbenchmark_obsjacobian( model, x )
%NLBENCHMARK_OBSJACOBIAN Summary of this function goes here
%   Detailed explanation goes here

H = model.alpha1 * model.alpha2 * x * (x^2)^(model.alpha2/2 - 1);
H(isnan(H)) = 0;

end

