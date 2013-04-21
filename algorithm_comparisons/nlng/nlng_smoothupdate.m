function [ state, weight ] = nlng_smoothupdate( display, algo, model, fh, obs, prev_state, weight)
%nlng_smoothupdate Apply a smooth update for the nonlinear non-Gaussian
%benchmark model.

% Set up integration schedule
ratio = 1.05;
num_steps = 250;
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
    else
        pf(ll+1).ancestor = 1:algo.N;
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
        
        if ~isinf(model.dfy)
            
            dfy = model.dfy;
            do = model.do;
            ds = model.ds-1;
            R = model.R;
            dy = obs - obs_mn;
            Rdy = R\dy;
            dyRdy = dy'*Rdy;
            tdist = 1+dyRdy/dfy;
            
%             % Calculate gradient and Hessian of the observation density
%             Dp = ((dfy+do)*H'*Rdy)/(dfy*tdist);
%             double_Rdy = [Rdy Rdy]';
%             double_Rdy = double_Rdy(:);
%             nasty_term = -H'*(R\H) + diag(double_Rdy);
%             D2p = ((dfy+do)*( (2/dfy)*H'*(Rdy*Rdy')*H/tdist + nasty_term ))/(dfy*tdist);
%             D2p = ((dfy+do)*( (2/dfy)*H'*(Rdy*Rdy')*H/tdist - H'*(R\H) ))/(dfy*tdist);
%             D2p = ((dfy+do)*( - H'*(R\H) ))/(dfy*tdist);
%             D2p = - H'*(R\H);
%             D2p = I;

            % Calculate value and gradient of the observation density
            p = mvnstpdf(obs', obs_mn', model.R, model.dfy);
            Dp_p = (model.dfy+model.do)*(H'/model.R)*(obs-obs_mn)/(model.dfy + (obs-obs_mn)'*(model.R\(obs-obs_mn)));
            
            % Match a Gaussian to these
            if p > 0
                [y, H, R] = gaussian_match_obs(x0, p, Dp_p, H);
            else
                y = obs; H = I; R = model.R;
            end
%             [y, ~] = gaussian_match_obs(x0, Dp, [], H, R);
%             y = obs;
            
        else
            
            R = model.R;
            y = obs - obs_mn + H*x0;
            
        end
        
        % Analytical flow
        [ x, wt_jac, prob_ratio] = linear_flow_move( lam, lam0, x0, m, P, y, H, R, algo.Dscale );
        
%         % Reverse transform
%         obs_mn_rev = nlng_h(model, x);
%         H_rev = nlng_obsjacobian(model, x);
%         p = mvnstpdf(obs', obs_mn_rev', model.R, model.dfy);
%         Dp_p = (model.dfy+model.do)*(H_rev'/model.R)*(obs-obs_mn_rev)/(model.dfy + (obs-obs_mn_rev)'*(model.R\(obs-obs_mn_rev)));
%         [y_rev, H_rev, R_rev] = gaussian_match_obs(x0, p, Dp_p, H_rev);
% %         R_rev = model.R;
% %         y_rev = obs - obs_mn_rev + H*x;
%         [ x0_rev, ~, ~] = linear_flow_move( lam0, lam, x, m, P, y_rev, H_rev, R_rev, algo.Dscale );
%         err = norm(x0 - x0_rev);
        
        % Store state
        state = [kk; x];
        pf(ll+1).state(:,ii) = state;
%         state_evo(:,ii,ll+1) = state;
        
        % Densities
        if ~isempty(prev_state)
            [~, trans_prob] = feval(fh.transition, model, prev_state(:,pf(ll+1).origin(ii)), state);
        else
            [~, trans_prob] = feval(fh.stateprior, model, state);
        end
        [~, lhood_prob] = feval(fh.observation, model, state, obs);
        prob(ii) = trans_prob + lam*lhood_prob;
        
        % Weight update
        if algo.Dscale == 0
            pf(ll+1).weight(ii) = pf(ll).weight(pf(ll+1).ancestor(ii)) + prob(ii) - last_prob(pf(ll+1).ancestor(ii)) + log(wt_jac);
        else
            pf(ll+1).weight(ii) = pf(ll).weight(pf(ll+1).ancestor(ii)) + prob(ii) - last_prob(pf(ll+1).ancestor(ii)) + prob_ratio;%log(wt_jac);%
        end
        
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
    figure(1), clf, hold on
    xlim([0 1]);
    for ii = 1:algo.N
        plot(lam_rng, squeeze(state_evo(2,ii,:)));
    end
    figure(2), clf, hold on
    for ii = 1:algo.N
        plot(squeeze(state_evo(2,ii,:)), squeeze(state_evo(3,ii,:)));
        plot(squeeze(state_evo(2,ii,end)), squeeze(state_evo(3,ii,end)), 'o');
    end
    figure(3), clf, hold on
    for ii = 1:algo.N
        plot(lam_rng, weight_evo(:,ii));
    end
    drawnow;
end

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
