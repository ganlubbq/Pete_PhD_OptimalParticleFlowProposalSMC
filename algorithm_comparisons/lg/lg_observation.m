function [ obs, prob ] = lg_observation( model, state, obs )
%lg_observation Sample and/or evaluate observation density for a linear
%Gaussian model.

% prob is a log-probability.

% Calculate observation mean
mn = model.H * state;

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

