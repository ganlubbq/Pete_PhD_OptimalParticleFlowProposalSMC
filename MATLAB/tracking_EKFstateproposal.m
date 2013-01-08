function [ state, prob ] = tracking_EKFstateproposal( model, prev_kk, prev_state, observ, state )
%NLBENCHMARK_STATEPROPOSAL Sample and/or calculate proposal density for
%2D tracking. This uses the EKF approximation to the OID.

% Predict state
prior_mn = model.A*prev_state;
prior_vr = model.Q;

% Point to linearise about
lin_state = prior_mn;

% Linearise
x = lin_state(1); y = lin_state(2); vx = lin_state(3); vy = lin_state(4);
rng_sq = x^2 + y^2;
rng = sqrt(rng_sq);
rng_32 = rng^3;
H = [-y/rng_sq               x/rng_sq                0      0    ; ...
     x/rng                   y/rng                   0      0    ; ...
     y*(vx*y - vy*x)/rng_32  x*(vy*x - vx*y)/rng_32  x/rng  y/rng ];


% EKF update
obs_mn = tracking_h(model,prior_mn);
[ppsl_mn, ppsl_vr] = ekf_update1(prior_mn, prior_vr, observ, H, model.R, obs_mn);

% Sample state if not provided
if (nargin<8)||isempty(state)
    state = mvnrnd(ppsl_mn', ppsl_vr)';
end

% Calculate probability if required
if nargout>1
    prob = loggausspdf(state, ppsl_mn, ppsl_vr);
else
    prob = [];
end

end

