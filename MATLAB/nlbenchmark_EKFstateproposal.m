function [ state, prob ] = nlbenchmark_EKFstateproposal( model, prev_kk, prev_state, observ, state )
%NLBENCHMARK_STATEPROPOSAL Sample and/or calculate proposal density for
%nonlinear benchmark. This uses the EKF approximation to the OID.

% Predict state
prior_mn = nlbenchmark_f(model, prev_kk, prev_state);
prior_vr = model.sigx;

% Point to linearise about
lin_state = prior_mn;

% Linearise
H = model.alpha1 * model.alpha2 * lin_state * (lin_state^2)^(model.alpha2/2 - 1);
H(isnan(H)) = 0;

% EKF update
obs_mn = nlbenchmark_h(model,prior_mn);
[ppsl_mn, ppsl_vr] = ekf_update1(prior_mn, prior_vr, observ, H, model.sigy, obs_mn);

% Sample state if not provided
if (nargin<8)||isempty(state)
    state = mvnrnd(ppsl_mn, ppsl_vr);
end

% Calculate probability if required
if nargout>1
    prob = loggausspdf(state, ppsl_mn, ppsl_vr);
else
    prob = [];
end

end

