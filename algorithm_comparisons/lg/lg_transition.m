function [ new_state, prob ] = lg_transition( model, state, new_state )
%lg_transition Sample and/or evaluate observation density for a linear
%Gaussian model.

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

