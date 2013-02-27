function [ state, prob ] = sinusoidseparation_EKFstateproposal( model, prev_kk, prev_state, observ, state )
% Sample and/or calculate proposal density for the sinusoidal separation 
%model. This uses the EKF approximation to the OID.

% Split state into continuous and discrete components
prev_con_state = prev_state(1:model.dsc,1);
prev_dis_state = prev_state(model.dsc+1:model.ds,1);

% Predict state
prior_mn = sinusoidseparation_f(model, prev_kk, prev_con_state);
prior_vr = model.Q;

% Point to linearise about
lin_state = prior_mn;

% Sample discrete state if not provided
if (nargin<8)||isempty(state)
    [dis_state, dis_ppsl_prob] = sinusoidseparation_discrete_transition(model, prev_dis_state);
else
    con_state = state(1:model.dsc,1);
    dis_state = state(model.dsc+1:model.ds,1);
end

% Linearise
H = sinusoidseparation_obsjacobian(model, lin_state, dis_state);

if dis_state(end)
    xi = chi2rnd(model.dfy);
    R = model.R / xi;
else
    R = model.R;
end

% EKF update
obs_mn = sinusoidseparation_h(model, prior_mn, dis_state);
[ppsl_mn, ppsl_vr] = ekf_update1(prior_mn, prior_vr, observ, H, R, obs_mn);
ppsl_vr = (ppsl_vr+ppsl_vr')/2;

% Sample continuous state if not provided
if (nargin<8)||isempty(state)
    con_state = mvnrnd(ppsl_mn', ppsl_vr)';
    state = [con_state; dis_state];
end

% Calculate probability if required
if nargout>1
    prob = dis_ppsl_prob + loggausspdf(con_state, ppsl_mn, ppsl_vr);
else
    prob = [];
end

end

