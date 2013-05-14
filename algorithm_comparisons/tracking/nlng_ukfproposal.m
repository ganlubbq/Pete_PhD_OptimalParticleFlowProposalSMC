function [state, ppsl_prob] = nlng_ukfproposal( model, prev_state, obs, state )
%nlng_ukfproposal Sample and/or evaluate UKF approximation of the OID for
% the nonlinear non-Gaussian benchmark model.

% If prev_state is empty, assume that this is the first step, so use the
% prior density instead of the transition density.

% Unpack the state
if ~isempty(prev_state)
    prev_kk = prev_state(1);
    prev_x = prev_state(2:end);
else
    prev_kk = 0;
end

% Prior
if isempty(prev_state)
    prior_mn = model.m1;
    prior_vr = model.P1;
else
    prior_mn = nlng_f(model, prev_kk, prev_x);
    prior_vr = model.Q;
end

% Sample a mixing value and scale the covariance
if ~isinf(model.dfy)
    xi = chi2rnd(model.dfy);
else
    xi = 1;
end
R = model.R / xi;

% UKF update
h = @(x, par)nlng_h(model, x);
[ppsl_mn, ppsl_vr] = ukf_update1(prior_mn, prior_vr, obs, h, R, [], [], [], 1);
ppsl_vr = 0.5*(ppsl_vr+ppsl_vr');

% Sample state if not provided
if (nargin<4)||isempty(state)
    kk = prev_kk + 1;
    x = mvnrnd(ppsl_mn', ppsl_vr)';
    state = [kk; x];
else
    kk = state(1);
    x = state(2:end);
    assert(kk==prev_kk+1);
end

% Calculate probability if required
if nargout>1
    ppsl_prob = loggausspdf(x, ppsl_mn, ppsl_vr);
else
    ppsl_prob = [];
end

end

