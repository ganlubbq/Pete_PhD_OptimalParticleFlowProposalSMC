function [ obs, prob ] = nlbenchmark_observation( model, state, obs )
%NLBENCHMARK_OBSERVATION Sample and/or calculate observation density for
%nonlinear benchmark.

% prob is a log-probability.

% Calculate observation mean
mn = nlbenchmark_h(model, state);

% Sample observation if not provided
if (nargin<3)||isempty(obs)
    obs = mvnrnd(mn, model.sigy)';
%     obs = mn + sqrt(model.sigy)*trnd(1);
end

% Calculate probability if required
if nargout>1
    prob = loggausspdf(obs, mn, model.sigy);
%     prob = log( tpdf( (obs-mn)/sqrt(model.sigy), 1) );
else
    prob = [];
end

end

