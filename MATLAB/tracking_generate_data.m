function [ time, state, observ ] = tracking_generate_data( model )
%NLBENCHMARK_GENERATE_DATA Generate data for a 2D tracking problem

% Initialise arrays
time = 1:model.K;
state = zeros(model.ds, model.K);
observ = zeros(model.do, model.K);

% First state
[state(:,1), ~] = tracking_stateprior(model);

% Loop through time
for kk = 1:model.K
    
    if kk > 1
        [state(:,kk), ~] = tracking_transition(model, kk-1, state(:,kk-1));
    end
    
    [observ(:,kk), ~] = tracking_observation(model, state(:,kk));
    
end

end

