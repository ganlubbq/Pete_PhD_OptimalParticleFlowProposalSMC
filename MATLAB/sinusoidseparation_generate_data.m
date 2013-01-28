function [ time, state, observ ] = sinusoidseparation_generate_data( model )
% Generate data according to the sinusoid separation model

% Initialise arrays
time = 1:model.K;
state = zeros(model.ds,model.K);
observ = zeros(model.do,model.K);

% First state
[state(:,1), ~] = sinusoidseparation_stateprior(model);
observ(:,1) = NaN;

% Loop through time
for kk = 2:model.K
    
    [state(:,kk), ~] = sinusoidseparation_transition(model, kk-1, state(:,kk-1));
    [observ(:,kk), ~] = sinusoidseparation_observation(model, state(:,kk));
    
end

end

