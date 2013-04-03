function [ new_state, prob ] = nlng_transition( model, state, new_state )
%nlng_transition Sample and/or evaluate observation density for the
%nonlinear non-Gaussian benchmark model.

% prob is a log-probability.
% state is a concatenation of the time index and the proper continuous state

% Unpack the state
kk = state(1);
x = state(2:end);

% Project to next time step
new_x_mn = nlng_f(model, kk, x);

% Sample state if not provided
if (nargin<3)||isempty(new_state)
    new_kk = kk + 1;
    new_x = mvnrnd(new_x_mn', model.Q)';
    new_state = [new_kk; new_x];
else
    new_kk = new_state(1);
    new_x = new_state(2:end);
    assert(new_kk==kk+1);
end

% Calculate probability if required
if nargout>1
    prob = loggausspdf(new_x, new_x_mn, model.Q);
else
    prob = [];
end

end

