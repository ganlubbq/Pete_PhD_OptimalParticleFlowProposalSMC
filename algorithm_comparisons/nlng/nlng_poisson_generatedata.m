function [ time, state, observ ] = nlng_poisson_generatedata( model )
%nlng_generatedata Generate a data set for the nonlinear non-Gaussian
%benchmark model.

% Initialise arrays
time = 1:model.K;
state = zeros(model.ds, model.K);
observ = zeros(model.do, model.K);

% First state
[state(:,1), ~] = nlng_stateprior(model);

% Loop through time
for kk = 1:model.K
    
    if kk > 1
        [state(:,kk), ~] = nlng_transition(model, state(:,kk-1));
    end
    
%     [observ(:,kk), ~] = nlng_observation(model, state(:,kk));
    [observ(:,kk), ~] = nlng_poisson_observation(model, state(:,kk));
    
end

end

