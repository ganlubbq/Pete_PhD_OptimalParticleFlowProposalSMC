function [ state, weight ] = drone_smoothupdate( display, algo, model, fh, obs, prev_state, weight)
%drone_smoothupdate Apply a smooth update for the drone model.

% Set up integration schedule
ratio = 1.2;
num_steps = 50;
% ratio = 1.1;
% num_steps = 100;
% ratio = 1.02;
% num_steps = 500;
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
    [~, init_lhood_prob(ii)] = feval(fh.observation, model, init_state(:,ii), obs);
end

% Sample mixing variable
init_mix = zeros(1,algo.N);
for ii = 1:algo.N
    if ~isinf(model.dfx)
        init_mix(ii) = chi2rnd(model.dfx);
    else
        init_mix(ii) = 1;
    end
end

% Initialise pf structure
pf(1).state = init_state;
pf(1).mix = init_mix;
pf(1).weight = init_weight;
pf(1).origin = 1:algo.N;

% Loop initialisation
last_prob = init_trans_prob;
last_full_prob = init_trans_prob + init_lhood_prob;

weight_diff = zeros(algo.N,L);
lhoods = zeros(algo.N,L);
errors = zeros(algo.N,L);
ess_evo = zeros(1,L);
bs_ess_evo = zeros(1,L);
bs_weight_evo = zeros(algo.N,L);
ess_evo(1) = calc_ESS(init_weight);

% Pseudo-time loop
for ll = 1:L-1
    
    % Pseudo-time
    lam0 = lam_rng(ll);
    lam = lam_rng(ll+1);
    
    % Initialise state and weight arrays
    pf(ll+1).state = zeros(model.ds, algo.N);
    pf(ll+1).mix = zeros(1, algo.N);
    pf(ll+1).weight = zeros(1, algo.N);
    prob = zeros(1, algo.N);
    full_prob = zeros(1, algo.N);
    
    % Resampling
    if algo.flag_intermediate_resample && (calc_ESS(pf(ll).weight)<(algo.N/2))
        pf(ll+1).ancestor = sample_weights(pf(ll).weight, algo.N, 2);
        flag_resamp = true;
    else
        pf(ll+1).ancestor = 1:algo.N;
        flag_resamp = false;
    end
    
    bs_weight = zeros(algo.N,1);
    
    % Particle loop
    for ii = 1:algo.N
        
        % Origin
        pf(ll+1).origin(ii) = pf(ll).origin(pf(ll+1).ancestor(ii));
        
        % Starting point
        x0 = pf(ll).state(:,pf(ll+1).ancestor(ii));
        xi0 = pf(ll).mix(1,pf(ll+1).ancestor(ii));
        
        % Observation mean
        obs_mn = drone_h(model, x0);
        
        % Linearise observation model around the current point
        R = model.R;
        H = drone_obsjacobian(model, x0);
        y = obs - obs_mn + H*x0;
        
        % Resolve bearing ambiguity        
        if y(1) > pi
            y(1) = y(1) - 2*pi;
        elseif y(1) < -pi
            y(1) = y(1) + 2*pi;
        end
        
        % Prior
        if isempty(prev_state)
            m = model.m1;
            P = model.P1;
        else
            m = model.A*prev_state(:,pf(ll+1).origin(ii));
            P = model.Q;
        end
        
        % Sample perturbation
        if algo.Dscale > 0
            zD = mvnrnd(zeros(model.ds,1)',eye(model.ds))';
        else
            zD = zeros(model.ds,1);
        end
        
        % SMoN scaling of transition density
        if ~isinf(model.dfx)
            xi = chi2rnd(model.dfx);
        else
            xi = 1;
        end
        P = P / xi;
        
%         %%%%%% TESTING 1ST ORDER TS MATCHING %%%%%%%%
%         
%         dx = x0 - m;
%         dfx = model.dfx;
%         ds = model.ds;
%         xP = (P\dx);
%         t_dist = 1 + xP'*dx/dfx;
%         HRH = H'*(R\H);
%         
%         prior_grad = -((dfx+ds)/dfx)*xP/t_dist;
%         prior_hess = ((dfx+ds)/dfx)*( - inv(P) + (2/dfx)*( xP*xP' )/t_dist )/t_dist;
%         
%         [hess_eigvec, hess_eigval] = eig(prior_hess);
%         hess_eigval(hess_eigval>0) = -1;
%         prior_hess = hess_eigvec*hess_eigval*hess_eigvec';
%         
% %         max_eig = max(eig(prior_hess));
% %         while ~isposdef(eye(ds) - lam*prior_hess\HRH)
% %             prior_hess = prior_hess - 1.1 * max_eig*eye(ds);
% %             fprintf(1,'.');
% %         end
%         
%         P = -inv(prior_hess);
%         m = x0 + P*prior_grad;
%         
% %         assert(isposdef(P));
%         
%         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         % Augmented flow
%         
%         dfx = model.dfx;
%         ds = model.ds;
%         Paug = inv( [xi0*inv(P), P\(x0-m); (x0-m)'/P, ((dfx+ds)/2-1)/xi0^2] );
%         [V,D] = eig(Paug);
%         D(D<0) = 1;
%         Paug = V*D*V';
%         maug = [x0; xi0] + Paug*[-xi0*(P\(x0-m)); ((dfx+ds)/2-1)/xi0-(1+(x0-m)'*(P\(x0-m)))/2];
%         Haug = [H, zeros(model.do,1)];
%         x0aug = [x0; xi0];
%         
%         % Analytical flow
%         [ xaug, prob_ratio, drift, diffuse] = linear_flow_move( lam, lam0, x0aug, maug, Paug, y, Haug, R, algo.Dscale, zD );
%         x = xaug(1:end-1);
%         xi = xaug(end);
%         
%         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%         %%%%%%%%%%%%%%%%%%%%%
%         %%% Second order matching %%%
%         T = drone_obssecondderivtensor(model, x0);
%         obs_diff = obs-obs_mn;
%         if obs_diff(1) > pi
%             obs_diff(1) = obs_diff(1) - 2*pi;
%         elseif obs_diff(1) < -pi
%             obs_diff(1) = obs_diff(1) + 2*pi;
%         end
%         yR = model.R\obs_diff;
%         invR = H'*(model.R\H);
%         for dd = 1:model.do
%             invR = invR - squeeze(T(dd,:,:))*yR(dd);
%         end
%         R = inv(invR);
%         y = x0 + invR\(H'*yR);
%         H = eye(model.ds);
%         %%%%%%%%%%%%%%%%%%%%%
        
        % Analytical flow
        [ x, prob_ratio, drift, diffuse] = linear_flow_move( lam, lam0, x0, m, P, y, H, R, algo.Dscale, zD );

        % Error estimate
        H_new = drone_obsjacobian(model, x);
        y_new = obs - drone_h(model, x) + H_new*x;
        [drift_new, diffuse_new] = linear_drift( lam, x, m, P, y_new, H_new, R, algo.Dscale );
        deter_err_est = 0.5*(lam-lam0)*(drift_new-drift);
        stoch_err_est = 0;%0.5*(diffuse_new-diffuse)*zD*sqrt(lam1-lam0);
        err_crit = deter_err_est'*deter_err_est + stoch_err_est'*stoch_err_est;
        errors(ii,ll+1) = err_crit;
        
        % Store state
        state = x;
        pf(ll+1).state(:,ii) = x;
        pf(ll+1).mix(1,ii) = xi;
        
        % Densities
        if ~isempty(prev_state)
            [~, trans_prob] = feval(fh.transition, model, prev_state(:,pf(ll+1).origin(ii)), state);
        else
            [~, trans_prob] = feval(fh.stateprior, model, state);
        end
        [~, lhood_prob] = feval(fh.observation, model, state, obs);
        prob(ii) = trans_prob + lam*lhood_prob;
        full_prob(ii) = trans_prob + lhood_prob;
        
        % Weight update
        if flag_resamp
            resamp_weight = 0;
        else
            resamp_weight = pf(ll).weight(pf(ll+1).ancestor(ii));
        end
        pf(ll+1).weight(ii) = resamp_weight + prob(ii) - last_prob(pf(ll+1).ancestor(ii)) + log(prob_ratio);
        
        lhoods(ii,ll+1) = lhood_prob + trans_prob;
        bs_weight(ii) = pf(ll+1).weight(ii) + (1-lam)*lhood_prob;
        bs_weight_evo(ii,ll+1) = bs_weight(ii);
        weight_diff(ii,ll+1) = full_prob(ii) - last_full_prob(pf(ll+1).ancestor(ii)) + log(prob_ratio);
        
        if ~isreal(pf(ll+1).weight(ii))
            pf(ll+1).weight(ii) = -inf;
        end
        
        assert(~isnan(pf(ll+1).weight(ii)));
        
    end
    
    last_prob = prob;
    last_full_prob = full_prob;
    
    ess_evo(ll+1) = calc_ESS(pf(ll+1).weight);
    bs_ess_evo(ll+1) = calc_ESS(bs_weight);
    
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
            plot(lam_rng, squeeze(state_evo(1,ii,:)), 'color', [0 rand rand]);
        end
        figure(2), clf, hold on
        for ii = 1:algo.N
            plot(squeeze(state_evo(1,ii,:)), squeeze(state_evo(2,ii,:)));
            plot(squeeze(state_evo(1,ii,end)), squeeze(state_evo(2,ii,end)), 'o');
        end
        figure(3), clf, hold on
        for ii = 1:algo.N
            plot(lam_rng, weight_evo(:,ii), 'color', [0 rand rand]);
        end
        figure(4), clf, hold on
        xlim([0 1]);
        for ii = 1:algo.N
            plot(lam_rng, mix_evo(:,ii), 'color', [0 rand rand]);
        end
        figure(5), clf, hold on
        plot(lam_rng, ess_evo);
        plot(lam_rng, bs_ess_evo, 'r');
        figure(6), clf, hold on
        plot(lam_rng, bs_weight_evo);
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

figure(7), plot(lam_rng, errors), drawnow

end

