function [state, ppsl_prob] = lg_oidproposal( model, prev_state, obs, state )
%lg_OIDproposal Sample and/or evaluate OID for a linear Gaussian model.

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

% KF update
[ppsl_mn, ppsl_vr] = kf_update(prior_mn, prior_vr, obs, model.H, model.R);

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

