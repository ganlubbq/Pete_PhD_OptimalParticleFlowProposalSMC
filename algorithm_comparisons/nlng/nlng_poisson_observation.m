function [ obs, prob ] = nlng_poisson_observation( model, state, obs )
%nlng_observation Sample and/or evaluate observation density for the
%nonlinear non-Gaussian benchmark model.

% prob is a log-probability.
% state is a concatenation of the time index and the proper continuous state

% Unpack the state
x = state(2:end);

% Project to observation space
obs_mn = nlng_h(model, x);

% Sample observation if not provided
if (nargin<3)||isempty(obs)
    obs = poissrnd(obs_mn);
end

% Calculate probability if required
if nargout>1
    prob = sum(log(poisspdf(obs, obs_mn)));
else
    prob = [];
end

end

