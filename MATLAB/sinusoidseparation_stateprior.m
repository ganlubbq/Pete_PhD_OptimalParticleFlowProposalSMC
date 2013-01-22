function [ state, prob ] = sinusoidseparation_stateprior( model, state )
% Sample and/or calculate state prior density for sinusoid separation
% model.

% prob is a log-probability.

% Sample state if not provided
if (nargin<2)||isempty(state)
    con_state = mvnrnd(model.x1_lin_mn', model.x1_lin_vr)';
    dis_state = binornd(1,[model.pprior_pres*ones(model.M,1); model.pprior_noise]);
    state = [con_state; dis_state];
else
    con_state = state(1:model.dsc,1);
    dis_state = state(model.dsc+1:model.ds,1);
end

% Calculate probability if required
if nargout>1
    prob = loggausspdf(con_state, model.x1_lin_mn, model.x1_lin_vr) ...
           +sum(log(binopdf(dis_state,1,[model.pprior_pres*ones(model.M,1); model.pprior_noise])));
else
    prob = [];
end

end

