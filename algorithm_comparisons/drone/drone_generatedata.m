function [ time, state, observ ] = drone_generatedata( model )
%drone_generatedata Generate a data set for the drone model.

% Initialise arrays
time = 1:model.K;
state = zeros(model.ds, model.K);
observ = zeros(model.do, model.K);

% First state
[state(:,1), ~] = drone_stateprior(model);

% Loop through time
for kk = 1:model.K
    
    if kk > 1
        [state(:,kk), ~] = drone_transition(model, state(:,kk-1));
    end
    
    [observ(:,kk), ~] = drone_observation(model, state(:,kk));
    
end

end

