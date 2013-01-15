function [ state, prob ] = linearGaussian_EKFstateproposal( model, prev_kk, prev_state, observ, state )
%NLBENCHMARK_STATEPROPOSAL Sample and/or calculate proposal density for
%2D tracking. This uses the EKF approximation to the OID.

% Predict state
prior_mn = model.A*prev_state;
prior_vr = model.Q;

% KF update
[ppsl_mn, ppsl_vr] = kf_update(prior_mn, prior_vr, observ, model.H, model.R);

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

