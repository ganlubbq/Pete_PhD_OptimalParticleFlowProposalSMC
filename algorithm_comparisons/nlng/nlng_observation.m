function [ obs, prob ] = nlng_observation( model, state, obs )
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

