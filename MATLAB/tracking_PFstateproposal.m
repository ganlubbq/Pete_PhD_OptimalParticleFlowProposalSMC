function [ state, prob ] = tracking_PFstateproposal( algo, model, prev_kk, prev_state, observ )
%NLBENCHMARK_STATEPROPOSAL Sample and/or calculate proposal density for
%2D tracking. This uses the particle flow approximation to the OID.

% State prior
prior_mn = model.A*prev_state;

% Sample it
[state, prob] = tracking_transition(model, prev_kk, prev_state);

% Step size
dl = 1/algo.L;

% Shorter variable names
Q = model.Q;
R = model.R;
m = prior_mn;
obs = observ;

% Flow integration loop
for ll = 1:algo.L
    
    % How far through?
    lam = (ll-1)*dl;
    
    % Linearise
    x = state(1); y = state(2); vx = state(3); vy = state(4);
    rng_sq = x^2 + y^2;
    rng = sqrt(rng_sq);
    rng_cb = rng^3;
    H = [-y/rng_sq               x/rng_sq                0      0    ; ...
         x/rng                   y/rng                   0      0    ; ...
         y*(vx*y - vy*x)/rng_cb  x*(vy*x - vx*y)/rng_cb  x/rng  y/rng ];
    
    % Find particle velocity using Gaussian approximation
    obs_mod = obs - tracking_h(model, state) + H*state;
    A = -0.5*Q*H'*((R+lam*H*Q*H')\H);
    b = (eye(model.ds)+2*lam*A)*((eye(model.ds)+lam*A)*Q*H'*(R\obs_mod)+A*m);
    v = A*state+b;
    
    % Find new state
    state = state + v*dl;
    
end

end

