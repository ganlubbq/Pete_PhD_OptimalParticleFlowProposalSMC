function [ state, prob ] = tracking_MHstateproposal( algo, model, prev_kk, prev_state, observ, state )
%NLBENCHMARK_STATEPROPOSAL Sample and/or calculate proposal density for
%2D tracking. This uses the MH approximation to the OID.

% Sample prior
prior_mn = model.A*prev_state;
prior_vr = model.Q;
[state, trans_prob] = tracking_transition(model, prev_kk, prev_state);
[~, lhood_prob] = tracking_observation(model, state, observ);
prob = trans_prob;

% Markov Chain
for ll = 1:algo.M
    
    % Linearise
    H = tracking_hessian(state);
    
    % EKF update
    obs_mn = tracking_h(model, state);
    [ppsl_mn, ppsl_vr] = ekf_update1(prior_mn, prior_vr, observ, H, model.R, obs_mn);
    
    % Propose a new state
    ppsl_state = mvnrnd(ppsl_mn', ppsl_vr)';
    ppsl_prob = loggausspdf(ppsl_state, ppsl_mn, ppsl_vr);
    [~, ppsl_trans_prob] = tracking_transition(model, prev_kk, prev_state, ppsl_state);
    [~, ppsl_lhood_prob] = tracking_observation(model, ppsl_state, observ);
    
    % Reverse proposal
    
    % Linearise
    H = tracking_hessian(state);
    
    % EKF update
    obs_mn = tracking_h(model, ppsl_state);
    [rev_ppsl_mn, rev_ppsl_vr] = ekf_update1(prior_mn, prior_vr, observ, H, model.R, obs_mn);
    
    rev_ppsl_prob = loggausspdf(state, rev_ppsl_mn, rev_ppsl_vr);
    
    % Accept
    ap = (ppsl_trans_prob+ppsl_lhood_prob) - (trans_prob+lhood_prob) + (rev_ppsl_prob-ppsl_prob);
    ap = min(0,ap);
    if log(rand) < ap
        state = ppsl_state;
        lhood_prob = ppsl_lhood_prob;
        trans_prob = ppsl_trans_prob;
        
        prob = prob + ap + ppsl_prob - rev_ppsl_prob;
    else
        prob = prob + log(1-exp(ap));
        
    end
    
end

end

