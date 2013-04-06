function [ new_state, prob ] = ha_transition( model, state, new_state )
%ha_transition Sample and/or evaluate observation density for the heartbeat
%alignment model.

% prob is a log-probability.

A = state(2);

% Sample state if not provided
if (nargin<3)||isempty(new_state)
    new_tau = model.tau_shift + gamrnd(model.tau_shape, model.tau_scale);
    new_A = mvnrnd(A, model.A1_vr);
    new_state = [new_tau; new_A];
else
    new_tau = new_state(1);
    new_A = new_state(2);
end

% Calculate probability if required
if nargout>1
    if new_tau-model.tau_shift > 0
        prob = loggausspdf(new_A, A, model.A1_vr) ...
            + log(gampdf(new_tau-model.tau_shift, model.tau_shape, model.tau_scale));
    else
        prob = -Inf;
    end
else
    prob = [];
end

end

