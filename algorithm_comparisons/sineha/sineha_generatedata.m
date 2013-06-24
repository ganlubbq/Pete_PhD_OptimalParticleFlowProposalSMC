function [ time, state, observ ] = sineha_generatedata( model )
%ha_generatedata Generate a data set for the heartbeat alignment model.

% Initialise arrays
time = 1:model.K;
state = zeros(model.ds, model.K);
observ = zeros(model.do, model.K);

% First state
[state(:,1), ~] = sineha_stateprior(model);

% Loop through time
for kk = 1:model.K
    
    if kk > 1
        [state(:,kk), ~] = sineha_transition(model, state(:,kk-1));
    end
    
    [observ(:,kk), ~] = sineha_observation(model, state(:,kk));
    
end

end

