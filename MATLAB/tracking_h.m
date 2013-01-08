function [ obs_mn ] = tracking_h( model, state )
%NLBENCHMARK_H Nonlinear function giving the mean of the current
%observation for 2D tracking.

obs_mn = zeros(model.do,1);

obs_mn(1) = atan2(state(2), state(1));
obs_mn(2) = magn(state(1:2));
obs_mn(3) = dot(state(1:2), state(3:4))/obs_mn(2);

end
