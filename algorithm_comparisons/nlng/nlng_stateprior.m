function [ state, prob ] = nlng_stateprior( model, state )
%nlng_stateprior Sample and/or evaluate observation density for the
%nonlinear non-Gaussian benchmark model.

% prob is a log-probability.
% state is a concatenation of the time index and the proper continuous state

% Sample state if not provided
if (nargin<2)||isempty(state)
    kk = 1;
    x = mvnrnd(model.m1', model.P1)';
    state = [kk; x];
else
    kk = state(1);
    x = state(2:end);
    assert(kk==1);
end

% Calculate probability if required
if nargout>1
    prob = loggausspdf(x, model.m1, model.P1);
else
    prob = [];
end

end

