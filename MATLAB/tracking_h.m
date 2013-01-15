function [ obs_mn ] = tracking_h( model, state )
%NLBENCHMARK_H Nonlinear function giving the mean of the current
%observation for 2D tracking.

obs_mn = zeros(model.do,1);

obs_mn(1) = atan2(state(2), state(1));
obs_mn(2) = magn(state(1:2));

v = state(4)*[cos(state(3)); sin(state(3))];
obs_mn(3) = dot(state(1:2), v)/obs_mn(2);

end
