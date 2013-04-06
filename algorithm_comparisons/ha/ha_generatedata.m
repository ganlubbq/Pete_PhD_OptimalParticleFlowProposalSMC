function [ time, state, observ ] = ha_generatedata( model )
%ha_generatedata Generate a data set for the heartbeat alignment model.

% Initialise arrays
time = 1:model.K;
state = zeros(model.ds, model.K);
observ = zeros(model.do, model.K);

% First state
[state(:,1), ~] = ha_stateprior(model);

% Loop through time
for kk = 1:model.K
    
    if kk > 1
        [state(:,kk), ~] = ha_transition(model, state(:,kk-1));
    end
    
    [observ(:,kk), ~] = ha_observation(model, state(:,kk));
    
end

end

