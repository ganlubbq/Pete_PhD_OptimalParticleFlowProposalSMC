function [ obs_mean ] = nlbenchmark_h( model, state )
%NLBENCHMARK_H Nonlinear function giving the mean of the current
%observation for the nonlinear benchmark

obs_mean = model.alpha1 * abs(state)^model.alpha2;

end
