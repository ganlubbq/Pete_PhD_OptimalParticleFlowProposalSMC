function [state, ppsl_prob] = drone_ekfproposal( model, prev_state, obs, state )
%drone_ekfproposal Sample and/or evaluate EKF approximation of the OID
% for the drone model.

% If prev_state is empty, assume that this is the first step, so use the
% prior density instead of the transition density.

% Prior
if isempty(prev_state)
    prior_mn = model.m1;
    prior_vr = model.P1;
else
    prior_mn = model.A*prev_state;
    prior_vr = model.Q;
end

% Sample a mixing value and scale the covariance
if ~isinf(model.dfx)
    xi = chi2rnd(model.dfx);
else
    xi = 1;
end
prior_vr = prior_vr / xi;

% Observation mean
obs_mn = drone_h(model, prior_mn);

% Linearise observation model around the prior mean
R = model.R;
H = drone_obsjacobian(model, prior_mn);
obs_lin = obs - obs_mn + H*prior_mn;

% Resolve bearing ambiguity
if obs_lin(1) > pi
    obs_lin(1) = obs_lin(1) - 2*pi;
elseif obs_lin(1) < -pi
    obs_lin(1) = obs_lin(1) + 2*pi;
end

% EKF update
[ppsl_mn, ppsl_vr] = ekf_update1(prior_mn, prior_vr, obs_lin, H, R, obs_mn);
ppsl_vr = (ppsl_vr + ppsl_vr')/2;

% Sample state if not provided
if (nargin<4)||isempty(state)
    state = mvnrnd(ppsl_mn', ppsl_vr)';
end

% Calculate probability if required
if nargout>1
    ppsl_prob = loggausspdf(state, ppsl_mn, ppsl_vr);
else
    ppsl_prob = [];
end

end

