function [ state, prob ] = lg_stateprior( model, state )
%lg_stateprior Sample and/or evaluate observation density for a linear
%Gaussian model.

% prob is a log-probability.

% Sample state if not provided
if (nargin<2)||isempty(state)
    state = mvnrnd(model.m1', model.P1)';
end

% Calculate probability if required
if nargout>1
    prob = loggausspdf(state, model.m1, model.P1);
else
    prob = [];
end

end

