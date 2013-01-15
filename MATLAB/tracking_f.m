function [ state_mn ] = tracking_f( model, state )
%NLBENCHMARK_F Nonlinear function giving the mean of the current
%state for 2D tracking.

state_mn = zeros(model.ds,1);

state_mn(1) = state(1) + state(4)*cos(state(3));
state_mn(2) = state(2) + state(4)*sin(state(3));
state_mn(3) = state(3);
state_mn(4) = max(state(4), model.min_speed);

end
