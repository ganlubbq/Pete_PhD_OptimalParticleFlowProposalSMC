function [state, ppsl_prob] = ha_ekfproposal( model, prev_state, obs, state )
%ha_ekfproposal Sample and/or evaluate EKF approximation of the OID for the
%heartbeat alignment model.

% If prev_state is empty, assume that this is the first step, so use the
% prior density instead of the transition density.

% Prior
if isempty(prev_state)
    A_mn = model.A1_mn;
    A_vr = model.A1_vr;
else
    A_mn = prev_state(2);
    A_vr = model.A_vr;
end
tau_mn = model.tau_shape*model.tau_scale + model.tau_shift;
tau_vr = model.tau_shape*model.tau_scale^2;
prior_mn = [tau_mn; A_mn];
prior_vr = diag([tau_vr, A_vr]);

% Observation mean
obs_mn = ha_h(model, prior_mn);

% Linearise observation model around the prior mean
H = ha_obsjacobian(model, prior_mn);
obs_lin = obs - obs_mn + H*prior_mn;

% Sample a mixing value and scale the covariance
if ~isinf(model.dfy)
    xi = chi2rnd(model.dfy);
else
    xi = 1;
end
R = model.R / xi;

% EKF update
[ppsl_mn, ppsl_vr] = ekf_update1(prior_mn, prior_vr, obs_lin, H, R, obs_mn);
ppsl_vr = (ppsl_vr+ppsl_vr')/2;

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

