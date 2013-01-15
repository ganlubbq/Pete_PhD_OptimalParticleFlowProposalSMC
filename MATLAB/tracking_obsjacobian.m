function [ H ] = tracking_obsjacobian( state )
%TRACKING_OBSJACOBIAN Calculate Jacobian for 2D tracking (position-velocity
%state and bearing, range, range-rate observation)

x = state(1); y = state(2); p = state(3); s = state(4);
rng_sq = x^2 + y^2;
rng = sqrt(rng_sq);
rng_3 = rng^3;
H = [-y/rng_sq               x/rng_sq                0      0    ; ...
     x/rng                   y/rng                   0      0    ; ...
     s*y*(y*cos(p) - x*sin(p))/rng_3  s*x*(x*sin(p) - y*cos(p))/rng_3  s*(y*cos(p) - x*sin(p))/rng  (x*cos(p) + y*sin(p))/rng ];

end

