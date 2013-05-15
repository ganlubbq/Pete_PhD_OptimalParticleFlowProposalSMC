function [ new_state, prob ] = tracking_transition( model, state, new_state )
%tracking_transition Sample and/or evaluate observation density for the
%tracking model.

% prob is a log-probability.

% Calculate new_state mean
mn = model.A * state;

% Sample state if not provided
if (nargin<3)||isempty(new_state)
    if isinf(model.dfx)
        new_state = mvnrnd(mn', model.Q)';
    else
        new_state = mvnstrnd(mn', model.Q, model.dfx)';
    end
end

% Calculate probability if required
if nargout>1
    if isinf(model.dfx)
        prob = loggausspdf(new_state, mn, model.Q);
    else
        prob = log(mvnstpdf(new_state', mn', model.Q, model.dfx));
    end
else
    prob = [];
end

end

