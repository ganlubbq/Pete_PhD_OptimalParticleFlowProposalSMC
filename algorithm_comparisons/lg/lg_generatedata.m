function [ time, state, observ ] = lg_generatedata( model )
%lg_generate_data Generate a data set for a linear Gaussian model.

% Initialise arrays
time = 1:model.K;
state = zeros(model.ds, model.K);
observ = zeros(model.do, model.K);

% First state
[state(:,1), ~] = lg_stateprior(model);

% Loop through time
for kk = 1:model.K
    
    if kk > 1
        [state(:,kk), ~] = lg_transition(model, state(:,kk-1));
    end
    
    [observ(:,kk), ~] = lg_observation(model, state(:,kk));
    
end

end

