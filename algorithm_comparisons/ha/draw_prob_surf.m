lam = 1;
A_rng = -1:0.1:1;
tau_rng = 0:0.01:2;

logprior = zeros(length(A_rng),length(tau_rng));
loglhood = zeros(length(A_rng),length(tau_rng));
logpost  = zeros(length(A_rng),length(tau_rng));

for AA = 1:length(A_rng)
    for tt = 1:length(tau_rng)
        
        x_At = [tau_rng(tt); A_rng(AA)];
        
        % Densities
        if ~isempty(prev_state)
            [~, trans_prob] = feval(fh.transition, model, prev_state, x_At);
        else
            [~, trans_prob] = feval(fh.stateprior, model, x_At);
        end
        [~, lhood_prob] = feval(fh.observation, model, x_At, obs);
        
        logprior(AA,tt) = trans_prob;
        loglhood(AA,tt) = lhood_prob;
        
    end
end


%%
lam = 1;
logpost = logprior+lam*loglhood;

 figure, surf(A_rng, tau_rng, logpost');
 shading interp;