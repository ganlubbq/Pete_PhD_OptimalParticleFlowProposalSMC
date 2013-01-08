function [ pf, ess, running_time ] = pf_standard( display, algo, model, fh, observ, flag_ppsl_type )
%PF_STANDARD Run a standard particle filter

tic;

if display.text
    fprintf(1, 'Particle Filtering.\n');
end

% Default proposal type
if nargin < 5
    flag_ppsl_type = 1;
end

if display.text
    fprintf(1, '   Time step 1.\n');
end

% Initialise arrays
pf = struct('state', cell(model.K,1), 'ancestor', cell(model.K,1), 'weight', cell(model.K,1));
pf(1).state = zeros(model.ds, algo.N);
pf(1).ancestor = zeros(1, algo.N);
pf(1).weight = zeros(1, algo.N);
ess = zeros(1,model.K);

% Sample first state from prior and calculate weight
for ii = 1:algo.N
    [pf(1).state(:,ii), ~] = feval(fh.stateprior, model);
    [~, pf(1).weight(1,ii)] = nlbenchmark_observation(model, pf(1).state(:,ii), observ(:,1));
end

ess(1) = calc_ESS(pf(1).weight);

pf(1).mn = mean(pf(1).state);
pf(1).sd = sqrt( sum( (pf(1).state-pf(1).mn).^2 )/algo.N );

% Loop through time
for kk = 2:model.K
    
    if display.text
        fprintf(1, '   Time step %u.\n', kk);
    end
    
    % Sample ancestors
    pf(kk).ancestor = sample_weights(pf(kk-1).weight, algo.N, 2);
    
    % Initialise state and weight arrays
    pf(kk).state = zeros(model.ds, algo.N);
    pf(kk).weight = zeros(1, algo.N);
    
    % Loop through particles
    for ii = 1:algo.N
        
        % Ancestory
        prev_state = pf(kk-1).state(:,pf(kk).ancestor(ii));
        
        % What type of proposal are we using?
        if flag_ppsl_type == 1
            %%% Bootstrap %%%
            
            % Sample a state and evaluate probabilities
            [state, trans_prob] = nlbenchmark_transition(model, kk-1, prev_state);
            [~, lhood_prob] = nlbenchmark_observation(model, state, observ(:,kk));
            ppsl_prob = trans_prob;
            
        elseif flag_ppsl_type == 2
            %%% EKF approx to OID %%%
            
            % Sample a state and evaluate probabilities
            [state, ppsl_prob] = nlbenchmark_EKFstateproposal(model, kk-1, prev_state, observ(:,kk));
            [~, trans_prob] = nlbenchmark_transition(model, kk-1, prev_state, state);
            [~, lhood_prob] = nlbenchmark_observation(model, state, observ(:,kk));
            
        elseif flag_ppsl_type == 3
            %%% Particle Flow OID %%%
            
            % Sample a state and evaluate probabilities
            [state, ppsl_prob] = nlbenchmark_PFstateproposal(algo, model, kk-1, prev_state, observ(:,kk));
            [~, trans_prob] = nlbenchmark_transition(model, kk-1, prev_state, state);
            [~, lhood_prob] = nlbenchmark_observation(model, state, observ(:,kk));
            
        else
            error('Invalid choice of proposal type');
        end
        
        pf(kk).state(:,ii) = state;
        pf(kk).weight(1,ii) = lhood_prob + trans_prob - ppsl_prob;
        
    end
    
    ess(kk) = calc_ESS(pf(kk).weight);
    
    norm_weight = exp(normalise_weights(pf(kk).weight));
    pf(kk).mn = sum(pf(kk).state*norm_weight');
    pf(kk).sd = sqrt( (pf(kk).state-pf(kk).mn).^2 * norm_weight' );
    
end

running_time = toc;

end

