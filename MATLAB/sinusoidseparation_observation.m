function [ obs, prob ] = sinusoidseparation_observation( model, state, obs )
% Sample and/or calculate observation density for sinusoidal separation
% model.

% prob is a log-probability.

% Split state into continuous and discrete components
con_state = state(1:model.dsc,1);
dis_state = state(model.dsc+1:model.ds,1);
obs_indic = state(model.ds,1);

% Calculate observation mean
mn = sinusoidseparation_h(model, con_state, dis_state);

% Sample observation if not provided
if (nargin<3)||isempty(obs)
    if obs_indic == 0
        obs = mvnrnd(mn', model.R)';
    elseif obs_indic == 1
        obs = mvcauchyrnd(mn', model.R)';
    end
end

% Calculate probability if required
if nargout>1
    if obs_indic == 0
        prob = loggausspdf(obs, mn, model.R);
    elseif obs_indic == 1
        prob = log(mvcauchypdf(obs', mn', model.R));
    end
else
    prob = [];
end

end

