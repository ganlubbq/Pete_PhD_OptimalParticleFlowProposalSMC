function [ H ] = tracking_hessian( state )
%TRACKING_HESSIAN Calculate Hessian for 2D tracking (position-velocity
%state and bearing, range, range-rate observation)

x = state(1); y = state(2); vx = state(3); vy = state(4);
rng_sq = x^2 + y^2;
rng = sqrt(rng_sq);
rng_32 = rng^3;
H = [-y/rng_sq               x/rng_sq                0      0    ; ...
     x/rng                   y/rng                   0      0    ; ...
     y*(vx*y - vy*x)/rng_32  x*(vy*x - vx*y)/rng_32  x/rng  y/rng ];

end

