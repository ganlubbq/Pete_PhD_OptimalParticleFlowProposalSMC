function [ state, prob ] = ha_stateprior( model, state )
%ha_stateprior Sample and/or evaluate observation density for the heartbeat
%alignment model.

% prob is a log-probability.

% Sample state if not provided
if (nargin<2)||isempty(state)
    tau = model.tau_shift + gamrnd(model.tau_shape, model.tau_scale);
    A = mvnrnd(model.A1_mn, model.A1_vr);
    state = [tau; A];
else
    tau = state(1);
    A = state(2);
end

% Calculate probability if required
if nargout>1
    if tau-model.tau_shift > 0
        prob = loggausspdf(A, model.A1_mn, model.A1_vr) ...
            + log(gampdf(tau-model.tau_shift, model.tau_shape, model.tau_scale));
    else
        prob = -Inf;
    end
else
    prob = [];
end

end

