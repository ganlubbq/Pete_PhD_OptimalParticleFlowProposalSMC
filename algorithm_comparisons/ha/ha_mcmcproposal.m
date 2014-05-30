function [state, prob] = ha_mcmcproposal( model, prev_state, obs )
%ha_mcmcproposal PF proposal using MCMC to target the OID.

M = 50;

% If prev_state is empty, assume that this is the first step, so use the
% prior density instead of the transition density.

if isempty(prev_state)
    [state, prob] = ha_stateprior(model);
else
    [state, prob] = ha_transition(model, prev_state);
end

% Parameters
K_vr = [0.02^2, 0; 0, 0.1^2];
target_step_size = 0.02;

% Initialise loop variables
trans_prob = prob;
lhood_prob = loggausspdf(obs, ha_h(model, state), model.R);
% MH loop
for mm = 1:M
    
%     % Simple
%     ppsl_mn = state;
%     ppsl_state = mvnrnd(ppsl_mn', K_vr)';
%     ppsl_prob = 0; rev_ppsl_prob = 0;
    
    % Calculate the log-likelihood gradient
    [loglhood_grad] = log_lhood_grad(model, obs, model.R, state);
    step_size = 0.003;%sqrt(2*target_step_size/max(abs(loglhood_grad)));%
    ppsl_mn = state + 0.5*step_size^2*loglhood_grad;

    % Propose a new state
    ppsl_state = mvnrnd(ppsl_mn', step_size^2*eye(model.ds))';
    ppsl_prob = loggausspdf(ppsl_state, ppsl_mn, step_size^2*eye(model.ds))';
    [rev_loglhood_grad] = log_lhood_grad(model, obs, model.R, ppsl_state);
    rev_ppsl_mn = ppsl_state + 0.5*step_size^2*rev_loglhood_grad;
    rev_ppsl_prob = loggausspdf(state, rev_ppsl_mn, step_size^2*eye(model.ds))';
    
    % Target density
    if isempty(prev_state)
        [~, ppsl_trans_prob] = ha_stateprior(model, ppsl_state);
    else
        [~, ppsl_trans_prob] = ha_transition(model, prev_state, ppsl_state);
    end
    ppsl_lhood_prob = loggausspdf(obs, ha_h(model, ppsl_state), model.R);
    
    % Acceptance probability
    ap = min(1, exp( (rev_ppsl_prob-ppsl_prob)+(ppsl_trans_prob+ppsl_lhood_prob)-(trans_prob+lhood_prob) ));
    
%     prob = prob + ppsl_prob-rev_ppsl_prob;
    
    if rand < ap
        state = ppsl_state;
        trans_prob = ppsl_trans_prob;
        lhood_prob = ppsl_lhood_prob;
        
%         prob = prob + log(ap);
        
    else
        prob = prob + log(1-ap);
        
    end
    
end

end

function [grad] = log_lhood_grad(model, y, R, x)
% Log-lhood with gradient and Hessian

h_x = ha_h(model, x);
H = ha_obsjacobian(model, x);
dy = y - h_x;

% % Function itself
% func = -(dy'/R)*dy/2;

% Gradient
grad = H'*(R\dy);

end

