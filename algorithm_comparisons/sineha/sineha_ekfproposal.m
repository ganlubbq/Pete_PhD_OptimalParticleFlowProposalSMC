function [state, ppsl_prob] = sineha_ekfproposal( model, prev_state, obs, state )
%ha_ekfproposal Sample and/or evaluate EKF approximation of the OID for the
%heartbeat alignment model.

% If prev_state is empty, assume that this is the first step, so use the
% prior density instead of the transition density.

% Prior
if isempty(prev_state)
    prior_mn = [model.A_shape*model.A_scale+model.A_shift;
                model.T1_mn;
                model.tau_shape*model.tau_scale;
                model.omega1_mn;
                model.phi1_mn;
                model.B1_mn];
else
    prior_mn = [model.A_shape*model.A_scale+model.A_shift;
                prev_state(2);
                model.tau_shape*model.tau_scale;
                prev_state(4:6)];
end
prior_vr = diag([model.A_shape*model.A_scale^2;
                 (exp(model.T_vol)-1)*exp(2*prior_mn(2)+model.T_vol);
                 model.tau_shape*model.tau_scale^2;
%                 prior_mn(3)^2/model.tau_shape;
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

