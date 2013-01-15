function [ new_state, prob ] = linearGaussian_transition( model, kk, state, new_state )
%NLBENCHMARK_TRANSITION Sample and/or calculate transition density for
%2D tracking.

% prob is a log-probability.

% Calculate new_state mean
mn = model.A * state;

% Sample state if not provided
if (nargin<4)||isempty(new_state)
    new_state = mvnrnd(mn', model.Q)';
end

% Calculate probability if required
if nargout>1
    prob = loggausspdf(new_state, mn, model.Q);
else
    prob = [];
end

end

