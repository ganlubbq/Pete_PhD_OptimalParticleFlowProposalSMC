function [ new_state, prob ] = sinusoidseparation_transition( model, kk, state, new_state )
% Sample and/or calculate transition density for sinusoidal separation
% model.

% state is the earlier state, which has time index kk. new_state is the
% following state. prob is a log-probability.

% Split state into continuous and discrete components
con_state = state(1:model.dsc,1);
dis_state = state(model.dsc+1:model.ds,1);

% Calculate new continuous state mean
mn = sinusoidseparation_f(model, kk, con_state);

% Sample state if not provided
if (nargin<4)||isempty(new_state)
    new_con_state = mvnrnd(mn', model.Q)';
    new_dis_state = sinusoidseparation_discrete_transition(model, dis_state);
    new_state = [new_con_state; new_dis_state];
else
    new_con_state = new_state(1:model.dsc,1);
    new_dis_state = new_state(model.dsc+1:model.ds,1);
end

% Calculate probability if required
if nargout>1
    [~, prob] = sinusoidseparation_discrete_transition(model, dis_state, new_dis_state);
    prob = prob + loggausspdf(new_con_state, mn, model.Q);
else
    prob = [];
end

end

