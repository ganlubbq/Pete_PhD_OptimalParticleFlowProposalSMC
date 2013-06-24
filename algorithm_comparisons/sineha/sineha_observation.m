function [ obs, prob ] = sineha_observation( model, state, obs )
%ha_observation Sample and/or evaluate observation density for the
%heartbeat alignment model.

% prob is a log-probability.

% Project to observation space
obs_mn = sineha_h(model, state);

% Sample observation if not provided
if (nargin<3)||isempty(obs)
    obs = mvnrnd(obs_mn', model.R)';
end

% Calculate probability if required
if nargout>1
    prob = loggausspdf(obs, obs_mn, model.R);
else
    prob = [];
end

end

