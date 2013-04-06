function [ obs, prob ] = ha_observation( model, state, obs )
%ha_observation Sample and/or evaluate observation density for the
%heartbeat alignment model.

% prob is a log-probability.

% Project to observation space
obs_mn = ha_h(model, state);

% Sample observation if not provided
if (nargin<3)||isempty(obs)
    if ~isinf(model.dfy)
        obs = mvnstrnd(obs_mn', model.R, model.dfy)';
    else
        obs = mvnrnd(obs_mn', model.R)';
    end
end

% Calculate probability if required
if nargout>1
    if ~isinf(model.dfy)
        prob = log(mvnstpdf(obs', obs_mn', model.R, model.dfy));
    else
        prob = loggausspdf(obs, obs_mn, model.R);
    end
else
    prob = [];
end

end

