function [ state, weight ] = nlng_smoothupdate( display, algo, model, fh, obs, prev_state, weight)
%nlng_smoothupdate Apply a smooth update for the nonlinear non-Gaussian
%benchmark model.

% Set up integration schedule
ratio = 1.2;
num_steps = 50;
scale_fact = (1-ratio)/(ratio*(1-ratio^num_steps));
lam_rng = cumsum([0 scale_fact*ratio.^(1:num_steps)]);
% lam_rng = 0:0.01:1;
L = length(lam_rng);

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

% % State evolution array (in case we want to plot the trajectories)
% state_evo = zeros(model.ds, algo.N, L+1);
% state_evo(:,:,1) = init_state;
% weight_evolution = zeros(L,algo.N);
% weight_evolution(1,:) = weight;

% Loop initialisation
last_prob = init_trans_prob;
% last_state_evo = state_evo;

% if display.plot_particle_paths
%     figure(1), clf, hold on
% end

errors = zeros(algo.N, L);

% Pseudo-time loop
for ll = 1:L-1
    
    % Pseudo-time
    lam0 = lam_rng(ll);
    lam = lam_rng(ll+1);
    
    % Initialise state and weight arrays
    pf(ll+1).state = zeros(model.ds, algo.N);
    pf(ll+1).weight = zeros(1, algo.N);
    prob = zeros(1, algo.N);
%     state_evo = zeros(model.ds, algo.N, L+1);
    
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
%         state_evo(:,ii,1:ll) = last_state_evo(:,pf(ll+1).ancestor(ii),1:ll);
        if isempty(prev_state)
            m = model.m1;
            P = model.P1;
        else
            prev_kk = prev_state(1);
            prev_x = prev_state(2:end,pf(ll+1).origin(ii));
            m = nlng_f(model, prev_kk, prev_x);
            P = model.Q;
        end
        
        % Starting point
        state = pf(ll).state(:,pf(ll+1).ancestor(ii));
        x0 = state(2:end);
        kk = state(1);
        
        % Observation mean
        obs_mn = nlng_h(model, x0);
        
        % Linearise observation model around the current point
        H = nlng_obsjacobian(model, x0);
        y = obs - obs_mn + H*x0;
        
        % SMoN scaling.
        if ~isinf(model.dfy)
            xi = chi2rnd(model.dfy);
        else
            xi = 1;
        end
        R = model.R / xi;
        
        % Sample perturbation
        if algo.Dscale > 0
            zD = mvnrnd(zeros(model.ds-1,1)',eye(model.ds-1))';
        else
            zD = zeros(model.ds-1,1);
        end
        
        % Analytical flow
        [ x, prob_ratio, drift, diffuse] = linear_flow_move( lam, lam0, x0, m, P, y, H, R, algo.Dscale, zD );
        
        % Error estimate
        H_new = nlng_obsjacobian(model, x);
        y_new = obs - nlng_h(model, x) + H_new*x;
        [drift_new, diffuse_new] = linear_drift( lam, x, m, P, y_new, H_new, R, algo.Dscale, zD );
        err_est = 0.5*(lam-lam0)*(drift_new-drift);% + 0.5*(diffuse_new-diffuse)*zD;
        err_crit = norm(err_est, 2);
        
        % Store state
        state = [kk; x];
        pf(ll+1).state(:,ii) = state;
%         state_evo(:,ii,ll+1) = state;
        errors(ii,ll+1) = err_crit;
        
        % Densities
        if ~isempty(prev_state)
            [~, trans_prob] = feval(fh.transition, model, prev_state(:,pf(ll+1).origin(ii)), state);
        else
            [~, trans_prob] = feval(fh.stateprior, model, state);
        end
        [~, lhood_prob] = feval(fh.observation, model, state, obs);
        prob(ii) = trans_prob + lam*lhood_prob;
        
        % Weight update
        if flag_resamp
            resamp_weight = 0;
        else
            resamp_weight = pf(ll).weight(pf(ll+1).ancestor(ii));
        end
        pf(ll+1).weight(ii) = resamp_weight + prob(ii) - last_prob(pf(ll+1).ancestor(ii)) + log(prob_ratio);
        
        if ~isreal(pf(ll+1).weight(ii))
            pf(ll+1).weight(ii) = -inf;
        end
        
        assert(~isnan(pf(ll+1).weight(ii)));
        
    end
    
    last_prob = prob;
%     last_state_evo = state_evo;
    
%     if display.plot_particle_paths && (rem(ll,10)==0)
%         figure(1), clf,
%         for ii = 1:algo.N
%             plot( squeeze(state_evo(2,:,1:ll+1))', squeeze(state_evo(3,:,1:ll+1))' );
%         end
%     end
    
end

state = pf(L).state;
weight = pf(L).weight;

state_evo = cat(3,pf.state);
weight_evo = cat(1,pf.weight);

% Plot particle paths (first state only)
if display.plot_particle_paths
    if ~algo.flag_intermediate_resample
        figure(1), clf, hold on
        xlim([0 1]);
        for ii = 1:algo.N
            plot(lam_rng, squeeze(state_evo(2,ii,:)), 'color', [0 rand rand]);
        end
        figure(2), clf, hold on
        for ii = 1:algo.N
            plot(squeeze(state_evo(2,ii,:)), squeeze(state_evo(3,ii,:)));
            plot(squeeze(state_evo(2,ii,end)), squeeze(state_evo(3,ii,end)), 'o');
        end
        figure(3), clf, hold on
        for ii = 1:algo.N
            plot(lam_rng, weight_evo(:,ii), 'color', [0 rand rand]);
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
            plot(lam_rng, weight_traj(ii,:), 'color', [0 rand rand]);
        end
        figure(2), clf, hold on
        for ii = 1:algo.N
            state_traj(:,ii,L) = state(2:3,ii);
            idx = ii;
            for ll = L-1:-1:1
                idx = pf(ll+1).ancestor(idx);
                state_traj(:,ii,ll) = pf(ll).state(2:3,idx);
            end
            plot(squeeze(state_traj(1,ii,:)), squeeze(state_traj(2,ii,:)), ':');
            plot(state_traj(1,ii,1), state_traj(2,ii,1), 'o');
            plot(state_traj(1,ii,end), state_traj(2,ii,end), 'x');
        end
        drawnow;
    end
end

figure(4), plot(lam_rng, errors'), drawnow;

end

% function v = v_iter(model, lam, x, obs, m, P)
% 
% H = nlng_obsjacobian(model, x);
% obs_mn = nlng_h(model, x);
% 
% if ~isinf(model.dfy)
%     
%     % Calculate value and gradient of the observation density
%     pdf = mvnstpdf(obs', obs_mn', model.R, model.dfy);
%     Dpdf_pdf = (model.dfy+model.do)*(H'/model.R)*(obs-obs_mn)/(model.dfy + (obs-obs_mn)'*(model.R\(obs-obs_mn)));
%     
%     % Match a Gaussian to these
%     [y, H, R] = gaussian_match_obs(x, pdf, Dpdf_pdf);
%     
% else
%     
%     R = model.R;
%     y = obs - obs_mn + H*x;
%     
% end
% 
% [A, b] = linear_flow(lam, m, P, y, H, R);
% v = A*x+b;
% 
% end
