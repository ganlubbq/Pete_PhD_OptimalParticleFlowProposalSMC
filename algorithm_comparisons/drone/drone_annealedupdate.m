function [ state, weight ] = drone_annealedupdate( display, algo, model, fh, obs, prev_state, weight)
%drone_annealedupdate Annealed particle filter update

% Select resampling times
resamp_times = [0 0.001 0.01 0.1 1];
L = length(resamp_times);

% Initialise particle filter structure
pf = repmat(struct('state', [], 'ancestor', [], 'weight', []), 1, L);

% Sample prior
init_weight = weight;
init_trans_prob = zeros(1, algo.N);
init_state = zeros(model.ds, algo.N);
for ii = 1:algo.N
    if ~isempty(prev_state)
        [init_state(:,ii), init_trans_prob(ii)] = feval(fh.transition, model, prev_state(:,ii));
    else
        [init_state(:,ii), init_trans_prob(ii)] = feval(fh.stateprior, model);
    end
end

% Initialise pf structure
pf(1).state = init_state;
pf(1).weight = init_weight;
pf(1).origin = 1:algo.N;

% Pseudo-time loop
for ll = 1:L-1
    
    % Pseudo-time
    lam0 = resamp_times(ll);
    lam = resamp_times(ll+1);
    
    % Initialise state and weight arrays
    pf(ll+1).state = zeros(model.ds, algo.N);
    pf(ll+1).mix = zeros(1, algo.N);
    pf(ll+1).weight = zeros(1, algo.N);
    
    % Resampling
    if algo.flag_intermediate_resample && (calc_ESS(pf(ll).weight)<(algo.N/2))
        pf(ll+1).ancestor = sample_weights(pf(ll).weight, algo.N, 2);
        flag_resamp = true;
    else
        pf(ll+1).ancestor = 1:algo.N;
        flag_resamp = false;
    end
    
    % Particle loop
    for ii = 1:algo.N
        
        % Origin
        pf(ll+1).origin(ii) = pf(ll).origin(pf(ll+1).ancestor(ii));
        
        % Starting point
        x = pf(ll).state(:,pf(ll+1).ancestor(ii));
        if ~isempty(prev_state)
            [~, trans_prob] = feval(fh.transition, model, prev_state(:,pf(ll+1).origin(ii)), x);
        else
            [~, trans_prob] = feval(fh.stateprior, model, x);
        end
        [~, lhood_prob] = feval(fh.observation, model, x, obs);
        old_post_prob = trans_prob + lam0*lhood_prob;
        
        % MCMC loop
        for mm = 1:10
            
            % MCMC proposal
            kernel_vr = chi2rnd(1)*eye(model.ds);
            x_ppsl = mvnrnd(x', kernel_vr)';
            
            % New probabilities
            if ~isempty(prev_state)
                [~, trans_prob] = feval(fh.transition, model, prev_state(:,pf(ll+1).origin(ii)), x_ppsl);
            else
                [~, trans_prob] = feval(fh.stateprior, model, x_ppsl);
            end
            [~, lhood_prob] = feval(fh.observation, model, x_ppsl, obs);
            new_post_prob = trans_prob + lam0*lhood_prob;
            
            % Accept?
            if log(rand) < new_post_prob-old_post_prob
                x = x_ppsl;
                old_post_prob = new_post_prob;
            end
            
        end
        
        % Store state
        state = x;
        pf(ll+1).state(:,ii) = x;
        
        % Densities
        [~, lhood_prob] = feval(fh.observation, model, state, obs);
        
        % Weight update
        if flag_resamp
            resamp_weight = 0;
        else
            resamp_weight = pf(ll).weight(pf(ll+1).ancestor(ii));
        end
        pf(ll+1).weight(ii) = resamp_weight + (lam-lam0)*lhood_prob;
        
        if ~isreal(pf(ll+1).weight(ii))
            pf(ll+1).weight(ii) = -inf;
        end
        
        assert(~isnan(pf(ll+1).weight(ii)));
        
    end
    
end

state = pf(L).state;
weight = pf(L).weight;

state_evo = cat(3,pf.state);
mix_evo = cat(1,pf.mix);
weight_evo = cat(1,pf.weight);

% Plot particle paths (first state only)
if display.plot_particle_paths
    if ~algo.flag_intermediate_resample
        figure(1), clf, hold on
        xlim([0 1]);
        for ii = 1:algo.N
            plot(resamp_times, squeeze(state_evo(1,ii,:)), 'color', [0 rand rand]);
        end
        figure(2), clf, hold on
        for ii = 1:algo.N
            plot(squeeze(state_evo(1,ii,:)), squeeze(state_evo(2,ii,:)));
            plot(squeeze(state_evo(1,ii,end)), squeeze(state_evo(2,ii,end)), 'o');
        end
        figure(3), clf, hold on
        for ii = 1:algo.N
            plot(resamp_times, weight_evo(:,ii), 'color', [0 rand rand]);
        end
        figure(4), clf, hold on
        xlim([0 1]);
        for ii = 1:algo.N
            plot(resamp_times, mix_evo(:,ii), 'color', [0 rand rand]);
        end
        drawnow;
    else
        weight_traj = zeros(algo.N,L);
        state_traj = zeros(2,algo.N,L);
        figure(1), clf, hold on
        xlim([0 1]);
        for ii = 1:algo.N
            weight_traj(ii,L) = weight(ii);
            idx = ii;
            for ll = L-1:-1:1
                idx = pf(ll+1).ancestor(idx);
                weight_traj(ii,ll) = pf(ll).weight(idx);
            end
            plot(resamp_times, weight_traj(ii,:), 'color', [0 rand rand]);
        end
        figure(2), clf, hold on
        for ii = 1:algo.N
            state_traj(:,ii,L) = state(1:2,ii);
            idx = ii;
            for ll = L-1:-1:1
                idx = pf(ll+1).ancestor(idx);
                state_traj(:,ii,ll) = pf(ll).state(1:2,idx);
            end
            plot(squeeze(state_traj(1,ii,:)), squeeze(state_traj(2,ii,:)), ':');
            plot(state_traj(1,ii,1), state_traj(2,ii,1), 'o');
            plot(state_traj(1,ii,end), state_traj(2,ii,end), 'x');
        end
        drawnow;
    end
end

end

