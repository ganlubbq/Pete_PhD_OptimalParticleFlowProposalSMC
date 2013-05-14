function [ obs, prob ] = tracking_observation( model, state, obs )
%tracking_observation Sample and/or evaluate observation density for the
%tracking model.

% prob is a log-probability.

% Calculate observation mean
mn = tracking_h(model, state);

% Sample observation if not provided
if (nargin<3)||isempty(obs)
    obs = mvnrnd(mn', model.R)';
end

% Calculate probability if required
if nargout>1
    prob = loggausspdf(obs, mn, model.R);
else
    prob = [];
end

end

