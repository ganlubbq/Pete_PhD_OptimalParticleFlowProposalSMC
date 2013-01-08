function [ obs, prob ] = tracking_observation( model, state, obs )
%NLBENCHMARK_OBSERVATION Sample and/or calculate observation density for
%2D tracking.

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

