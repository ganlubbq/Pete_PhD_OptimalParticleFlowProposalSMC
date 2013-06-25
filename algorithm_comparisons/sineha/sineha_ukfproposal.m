function [state, ppsl_prob] = sineha_ukfproposal( model, prev_state, obs, state )
%ha_ukfproposal Sample and/or evaluate UKF approximation of the OID for
%the heartbeat alignment model.

% If prev_state is empty, assume that this is the first step, so use the
% prior density instead of the transition density.

% Prior
if isempty(prev_state)
    prior_mn = [model.A_shape*model.A_scale+model.A_shift;
                model.T1_mn;
                model.tau_mn;
                model.omega1_mn;
                model.phi1_mn;
                model.B1_mn];
else
    prior_mn = [model.A_shape*model.A_scale+model.A_shift;
                prev_state(2);
                model.tau_mn;
                prev_state(4:6)];
end
prior_vr = diag([model.A_shape*model.A_scale^2;
                 (exp(model.T_vol)-1)*exp(2*prior_mn(2)+model.T_vol);
%                  (prior_mn(3)-prior_mn(2))^2/model.tau_shape;
                prior_mn(3)^2/model.tau_shape;
                 model.omega_vr;
                 model.phi_vr;
                 model.B_vr]);

R = model.R;

% UKF update
h = @(x, par)sineha_h(model, x);
[ppsl_mn, ppsl_vr] = ukf_update1(prior_mn, prior_vr, obs, h, R, [], [], [], 1);
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

