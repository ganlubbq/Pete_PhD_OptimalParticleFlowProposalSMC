function [ state, prob ] = sinusoidseparation_UKFstateproposal( model, prev_kk, prev_state, observ, state )
% Sample and/or calculate proposal density for the sinusoidal separation 
%model. This uses the UKF approximation to the OID.

% Split state into continuous and discrete components
prev_con_state = prev_state(1:model.dsc,1);
prev_dis_state = prev_state(model.dsc+1:model.ds,1);

% Predict state
prior_mn = sinusoidseparation_f(model, prev_kk, prev_con_state);
prior_vr = model.Q;

% Sample discrete state if not provided
if (nargin<8)||isempty(state)
    [dis_state, dis_ppsl_prob] = sinusoidseparation_discrete_transition(model, prev_dis_state);
else
    con_state = state(1:model.dsc,1);
    dis_state = state(model.dsc+1:model.ds,1);
end

if dis_state(end)
    xi = exprnd(0.5);
    R = model.R / xi;
end

% UKF update
h = @(x, par)sinusoidseparation_h(model, x, dis_state);
[ppsl_mn, ppsl_vr] = ukf_update1(prior_mn, prior_vr, observ, h, R, [], [], [], 1);
ppsl_vr = 0.5*(ppsl_vr+ppsl_vr');

% Sample state if not provided
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

