function [ obs_mn ] = nlbenchmark_h( model, state )
%NLBENCHMARK_H Nonlinear function giving the mean of the current
%observation for the nonlinear benchmark

nl_mn = model.alpha1 * abs(state).^model.alpha2;
obs_mn = model.Hlin * nl_mn;

end
