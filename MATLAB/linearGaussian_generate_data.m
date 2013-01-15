function [ time, state, observ ] = linearGaussian_generate_data( model )
%NLBENCHMARK_GENERATE_DATA Generate data for a 2D tracking problem

% Initialise arrays
time = 1:model.K;
state = zeros(model.ds, model.K);
observ = zeros(model.do, model.K);

% First state
[state(:,1), ~] = linearGaussian_stateprior(model);

% Loop through time
for kk = 1:model.K
    
    if kk > 1
        [state(:,kk), ~] = linearGaussian_transition(model, kk-1, state(:,kk-1));
    end
    
    [observ(:,kk), ~] = linearGaussian_observation(model, state(:,kk));
    
end

end

