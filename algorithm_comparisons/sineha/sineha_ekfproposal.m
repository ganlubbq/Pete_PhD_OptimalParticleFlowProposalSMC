function [state, ppsl_prob] = sineha_ekfproposal( model, prev_state, obs, state )
%ha_ekfproposal Sample and/or evaluate EKF approximation of the OID for the
%heartbeat alignment model.

% If prev_state is empty, assume that this is the first step, so use the
% prior density instead of the transition density.

% Prior
if isempty(prev_state)
    prior_mn = [model.A1_mn; model.T1_mn; model.tau1_mn; model.omega1_mn; model.phi1_mn; model.B1_mn];
else
    prior_mn = prev_state;
end
prior_vr = diag([prior_mn(1)^2/model.A_shape;
                 prior_mn(2)^2/model.T_shape;
                 prior_mn(3)^2/model.tau_shape;
                 model.omega_vr;
                 model.phi_vr;
                 model.B_vr]);
        
% Observation mean
obs_mn = sineha_h(model, prior_mn);

% Linearise observation model around the prior mean
H = sineha_obsjacobian(model, prior_mn);
obs_lin = obs - obs_mn + H*prior_mn;
R = model.R;

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

