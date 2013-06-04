function [ state, weight ] = ha_smoothupdate( display, algo, model, fh, obs, prev_state, weight)
%ha_smoothupdate Apply a smooth update for the heartbeat alignment model.

% Set up integration schedule
ratio = 1.2;
num_steps = 50;
scale_fact = (1-ratio)/(ratio*(1-ratio^num_steps));
lam_rng = cumsum([0 scale_fact*ratio.^(1:num_steps)]);
% lam_rng = 0:0.001:1;
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

% Loop initialisation
last_prob = init_trans_prob;

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
    
    % Resampling
    if algo.flag_intermediate_resample && (calc_ESS(pf(ll).weight)<(0.5*algo.N))
        pf(ll+1).ancestor = sample_weights(pf(ll).weight, algo.N, 2);
    else
        pf(ll+1).ancestor = 1:algo.N;
    end
    
    % Particle loop
    for ii = 1:algo.N
        
        % Origin
        pf(ll+1).origin(ii) = pf(ll).origin(pf(ll+1).ancestor(ii));
        if isempty(prev_state)
            A_mn = model.A1_mn;
            A_vr = model.A1_vr;
        else
            A_mn = prev_state(2,pf(ll+1).origin(ii));
            A_vr = model.A_vr;
        end
        
        % Starting point
        state = pf(ll).state(:,pf(ll+1).ancestor(ii));
        x0 = state(1:end);
        
        % Observation mean
        obs_mn = ha_h(model, x0);
        
        % Linearise observation model around the current point
        H = ha_obsjacobian(model, x0);
        y = obs - obs_mn + H*x0;
        
        % SMoN scaling.
        if ~isinf(model.dfy)
            xi = chi2rnd(model.dfy);
        else
            xi = 1;
        end
        R = model.R / xi;
        
%         % Flow matching
%         tau = x0(1);
%         prior_grad = (model.tau_shape-1)./(tau-model.tau_shift) - 1/model.tau_scale;
%         prior_hess = -(model.tau_shape-1)./(tau-model.tau_shift)^2;
%         tau_vr = -1/prior_hess;
%         tau_mn = tau + tau_vr*prior_grad;
        
        % Calculate value and gradient of the tau prior density
        tau = x0(1);
        pdf = gampdf(tau-model.tau_shift, model.tau_shape, model.tau_scale);
        Dpdf_pdf = (model.tau_shape-1)/(tau-model.tau_shift) - 1/model.tau_scale;
        
        % Match a Gaussian to these
        if (tau > model.tau_shift) && (pdf > 0)
            [tau_mn, tau_vr] = gaussian_match_prior(tau, pdf, Dpdf_pdf);
        else
            tau_mn = model.tau_shape * model.tau_scale;
            tau_vr = model.tau_shape * model.tau_scale^2;
        end
        
        % Matched prior Gaussian
        m = [tau_mn; A_mn];
        P = diag([tau_vr, A_vr]);
        
        % Analytical flow
        [ x, wt_jac, prob_ratio] = linear_flow_move( lam, lam0, x0, m, P, y, H, R, algo.Dscale );
        
%         % Reverse transform
%         obs_mn_rev = ha_h(model, x);
%         H_rev = ha_obsjacobian(model, x);
%         R_rev = R;
%         y_rev = obs - obs_mn_rev + H*x;
%         tau = x(1);
%         pdf = gampdf(tau-model.tau_shift, model.tau_shape, model.tau_scale);
%         Dpdf_pdf = (model.tau_shape-1)/(tau-model.tau_shift) - 1/model.tau_scale;
%         if (tau > model.tau_shift) && (pdf > 0)
%             [tau_mn, tau_vr] = gaussian_match_prior(tau, pdf, Dpdf_pdf);
%         else
%             tau_mn = model.tau_shape * model.tau_scale;
%             tau_vr = model.tau_shape * model.tau_scale^2;
%         end
%         m = [tau_mn; A_mn];
%         P = diag([tau_vr, A_vr]);
%         [ x0_rev, ~, ~] = linear_flow_move( lam0, lam, x, m, P, y_rev, H_rev, R_rev, algo.Dscale );
%         errors(ii,ll+1) = norm(x0 - x0_rev);

%         % Reverse transform
%         obs_mn_rev = ha_h(model, x);
%         H_rev = ha_obsjacobian(model, x);
%         R_rev = R;
%         y_rev = obs - obs_mn_rev + H*x;
%         tau = x(1);
%         prior_grad = (model.tau_shape-1)./(tau-model.tau_shift) - 1/model.tau_scale;
%         prior_hess = -(model.tau_shape-1)./(tau-model.tau_shift)^2;
%         tau_vr = -1/prior_hess;
%         tau_mn = tau + tau_vr*prior_grad;
%         m = [tau_mn; A_mn];
%         P = diag([tau_vr, A_vr]);
%         [ x0_rev, ~, ~] = linear_flow_move( lam0, lam, x, m, P, y_rev, H_rev, R_rev, algo.Dscale );
%         errors(ii,ll+1) = norm(x0 - x0_rev);
        
        % Store state
        pf(ll+1).state(:,ii) = x;
        
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
        if isinf(last_prob(pf(ll+1).ancestor(ii)))
            pf(ll+1).weight(ii) = -inf;
        end
        
        assert(~isnan(pf(ll+1).weight(ii)));
        
    end
    
    last_prob = prob;
    
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
        plot(squeeze(state_evo(1,ii,:)), squeeze(state_evo(2,ii,:)));
        plot(squeeze(state_evo(1,ii,end)), squeeze(state_evo(2,ii,end)), 'o');
    end
    figure(3), clf, hold on
    for ii = 1:algo.N
        plot(lam_rng, weight_evo(:,ii));
    end
    drawnow;
end

figure(4), plot(errors'), drawnow;

end







% function [ state, weight ] = ha_smoothupdate( display, algo, model, fh, obs, prev_state, weight)
% %ha_smoothupdate Apply a smooth update for the heartbeat alignment model.
% 
% % Define resampling points
% lam_resam = [0 1E-5 1E-4 1E-3 1E-2 1E-1  0.5 1];
% % lam_resam = [0 1];
% 
% % Initialise particle filter structure
% pf = repmat(struct('state', [], 'ancestor', [], 'weight', []), 1, length(lam_resam));
% 
% % Sample prior
% init_weight = weight;
% init_trans_prob = zeros(1, algo.N);
% init_state = zeros(model.ds, algo.N);
% for ii = 1:algo.N
%     if ~isempty(prev_state)
%         [init_state(:,ii), init_trans_prob(ii)] = feval(fh.transition, model, prev_state(:,ii));
%     else
%         [init_state(:,ii), init_trans_prob(ii)] = feval(fh.stateprior, model);
%     end
% end
% 
% last_prob = init_trans_prob;
% 
% pf(1).state = init_state;
% pf(1).weight = init_weight;
% pf(1).origin = 1:algo.N;
% 
% % Set up for plotting
% if display.plot_particle_paths
%     figure(1); clf; hold on;
% end
% 
% % Loop over resampling intervals
% for ll = 1:length(lam_resam)-1
%     
%     % Timing
%     start_lam = lam_resam(ll);
%     stop_lam = lam_resam(ll+1);
%     
%     % Initialise state and weight arrays
%     pf(ll+1).state = zeros(model.ds, algo.N);
%     pf(ll+1).weight = zeros(1, algo.N);
%     prob = zeros(1, algo.N);
%     
%     % Resampling
%     if algo.flag_intermediate_resample
%         pf(ll+1).ancestor = sample_weights(pf(ll).weight, algo.N, 2);
%         last_weights = zeros(1,algo.N);
%     else
%         pf(ll+1).ancestor = 1:algo.N;
%         last_weights = pf(ll).weight;
%     end
%     
%     % Loop over particles
%     for ii = 1:algo.N
%         
%         % Origin
%         pf(ll+1).origin(ii) = pf(ll).origin(pf(ll+1).ancestor(ii));
%         if isempty(prev_state)
%             A_mn = model.A1_mn;
%             A_vr = model.A1_vr;
%         else
%             A_mn = prev_state(2,pf(ll+1).origin(ii));
%             A_vr = model.A_vr;
%         end
%         
%         % Starting point
%         start_x = pf(ll).state(:,pf(ll+1).ancestor(ii));
%         
%         % Numerically integrate over the interval
%         [lam_evo, x_evo, jac_evo] = RK45int(start_lam, stop_lam, start_x, model, obs, A_mn, A_vr, D);
%         
%         % Store state
%         state = x_evo(:,end);
%         pf(ll+1).state(:,ii) = state;
%         
%         % Densities
%         if ~isempty(prev_state)
%             [~, trans_prob] = feval(fh.transition, model, prev_state(:,pf(ll+1).origin(ii)), state);
%         else
%             [~, trans_prob] = feval(fh.stateprior, model, state);
%         end
%         [~, lhood_prob] = feval(fh.observation, model, state, obs);
%         prob(ii) = trans_prob + stop_lam*lhood_prob;
%         
%         % Weight update
%         if isreal(log(jac_evo(end)))
%             pf(ll+1).weight(ii) = last_weights(pf(ll+1).ancestor(ii)) + log(jac_evo(end)) + prob(ii) - last_prob(pf(ll+1).ancestor(ii));
% %             pf(ll+1).weight(ii) = last_weights(pf(ll+1).ancestor(ii)) + prob(ii) - last_prob(pf(ll+1).ancestor(ii));
%         else
%             pf(ll+1).weight(ii) = -inf;
%         end
%         
%         if isnan(pf(ll+1).weight(ii))
%             pf(ll+1).weight(ii) = -inf;
%         end
%         
%         % Plot
%         if display.plot_particle_paths
%             plot(x_evo(1,:), x_evo(2,:));
%             if ll == length(lam_resam)-1
%                 plot(state(1), state(2), 'o');
%             end
%             drawnow;
%         end
%         
%     end
%     
%     last_prob = prob;
%     
% end
% 
% state = pf(end).state;
% weight = pf(end).weight;
% 
% end

function [lam_evo, x_evo, jac_evo] = RK45int(lam0, lamf, x0, model, obs, A_mn, A_vr, D)

% Parameters
int_lam_coef = [1/5, 3/10, 4/5, 8/9, 1, 1];
int_x_coef = [
    1/5         3/40    44/45   19372/6561      9017/3168       35/384
    0           9/40    -56/15  -25360/2187     -355/33         0
    0           0       32/9    64448/6561      46732/5247      500/1113
    0           0       0       -212/729        49/176          125/192
    0           0       0       0               -5103/18656     -2187/6784
    0           0       0       0               0               11/84
    0           0       0       0               0               0
    ];
err_coef = [71/57600; 0; -71/16695; 71/1920; -17253/339200; 22/525; -1/40];
hmin = 1E-6;
hmax = 1E-1;
pow = 0.2;
tol = 1E-3;
threshold = 1E-3;

% arrays
lam_evo = zeros(1,0);
x_evo = zeros(model.ds, 0);
jac_evo = zeros(1,0);

% Setup
lam = lam0;
x = x0;
jac = 1;
[f, A, ~, D] = f_iter(model, lam0, x0, obs, A_mn, A_vr, D);

% Store
lam_evo = [lam_evo lam];
x_evo = [x_evo x];
jac_evo = [jac_evo jac];

% choose initial step size
h = min(hmax, lamf-lam0);
rh = norm(f ./ max(abs(x),threshold),inf) / (0.8 * tol^pow);
if h * rh > 1
    h = 1 / rh;
end
h = max(h, hmin);

no_fails = true;

% Main loop
while lam < lamf
    
    % Are we nearly there?
    if lam + 1.1*h > lamf
        h = lamf - lam;
    end
    
    % Calculate flows at intermediate points
    f_arr = zeros(model.ds,7);
    trA_arr = zeros(1,7);
%     f_arr(:,1) = f;
%     trA_arr(1) = trace(A);
    [f_arr(:,1), A, ~] = f_iter(model, lam, x, obs, A_mn, A_vr, D);
    trA_arr(1) = trace(A);
    for ii = 1:5
        lam_int = lam + h*int_lam_coef(ii);
        x_int = x + h*f_arr*int_x_coef(:,ii);
        [f_arr(:,ii+1), A, ~] = f_iter(model, lam_int, x_int, obs, A_mn, A_vr, D);
        trA_arr(ii+1) = trace(A);
    end
    
    % New state
    lam_new = lam + h;
    x_new = x + h*f_arr*int_x_coef(:,6);
    %%% Stochastic bit - BODGE! %%%
    if ~all(D(:)==0)
        x_new = mvnrnd(x_new', 2*h*D)';
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [f_arr(:,7), A, ~] = f_iter(model, lam_new, x_new, obs, A_mn, A_vr, D);
    
    % Jacobian update
    jac_new = jac * (1+h*trA_arr*int_x_coef(:,6));
    
    % Step size control
    err = h * norm((f_arr * err_coef) ./ max(max(abs(x),abs(x_new)),threshold),inf); 
    if err > tol
        % failed - no update, decrease step size
        no_fails = false;
        h = max(hmin, h * max(0.1, 0.8*(tol/err)^pow));
    else
        
        % suceeded - update
        lam = lam + h;
        x = x_new;
        jac = jac_new;
        f = f_arr(:,7);
        
        if no_fails
            % Increase step size
            temp = 1.25*(err/tol)^pow;
            if temp > 0.2
                h = h / temp;
            else
                h = 5.0*h;
            end
        end
        
        no_fails = true;
        
        % Store
        lam_evo = [lam_evo lam];
        x_evo = [x_evo x];
        jac_evo = [jac_evo jac];
        
    end
    
end

end







% % Set up integration schedule
% % num_steps = 200;ratio = 1.03;
% num_steps = 100;ratio = 1.07;
% scale_fact = (1-ratio)/(ratio*(1-ratio^num_steps));
% lam_rng = cumsum([0 scale_fact*ratio.^(1:num_steps)]);
% % lam_rng = [1E-5:1E-5:1E-4 2E-4:1E-4:1E-3 2E-3:1E-3:1E-2 2E-2:1E-2:1E-1 2E-1:1E-1:9E-1 9.1E-1:1E-2:1];
% L = length(lam_rng);
% 
% 
% % State evolution array (in case we want to plot the trajectories)
% state_evolution = zeros(model.ds, algo.N, L);
% state_evolution(:,:,1) = state;
% weight_evolution = zeros(L,algo.N);
% weight_evolution(1,:) = weight;
% 
% % Weight arrays
% inc_weight = zeros(1, algo.N);
% prob = zeros(1, algo.N);
% last_prob = init_trans_prob;

% for ii = 1:algo.N
%     
%     x = state(:,ii);
%     A_mn = A_mn_arr(1,ii);
%     
%     lam_rng = [0 1];
%     [lam, x] = ode23(@(lam_in, x_in) v_iter(model, lam_in, x_in, obs, A_mn, A_vr), lam_rng, x);
%     state(:,ii) = x(end,:)';
%     
%     % Densities
%     if ~isempty(prev_state)
%         [~, trans_prob] = feval(fh.transition, model, prev_state(:,ii), state(:,ii));
%     else
%         [~, trans_prob] = feval(fh.stateprior, model, state(:,ii));
%     end
%     [~, lhood_prob] = feval(fh.observation, model, state(:,ii), obs);
%     
%     % Update weight
%     weight(ii) = weight(ii) + trans_prob + lhood_prob - init_trans_prob(ii);
%     if ~isreal(weight(ii))||isnan(weight(ii))
%         weight(ii) = -inf;
%     end
%     
%     if display.plot_particle_paths
%         plot(x(:,1), x(:,2));
%         plot(x(end,1), x(end,2), 'o');
%     end
%     
% end



% % Pseudo-time loop
% for ll = 1:L-1
%     
%     % Resampling
%     
%     % Pseudo-time
%     lam = lam_rng(ll);
%     dl = lam_rng(ll+1)-lam_rng(ll);
%     
%     % Particle loop
%     for ii = 1:algo.N
%         
%         % Get state
%         x = state(:,ii);
%         A_mn = A_mn_arr(1,ii);
%         
%         % Unpack
%         tau = x(1);
%         A = x(2);
%         
%         % Calculate value and gradient of the tau prior density
%         pdf = gampdf(tau-model.tau_shift, model.tau_shape, model.tau_scale);
%         Dpdf_pdf = (model.tau_shape-1)/(tau-model.tau_shift) - 1/model.tau_scale;
% 
%         % Match a Gaussian to these
%         [tau_mn, tau_vr] = gaussian_match_prior(tau, pdf, Dpdf_pdf);
%         
%         % Matched prior Gaussian
%         m = [tau_mn; A_mn];
%         P = diag([tau_vr, A_vr]);
%         
%         % Observation mean
%         obs_mn = ha_h(model, x);
%         
%         % Linearise observation model around the current point
%         H = ha_obsjacobian(model, x);
%         
%         if ~isinf(model.dfy)
%             
%             % Calculate value and gradient of the observation density
%             pdf = mvnstpdf(obs', obs_mn', model.R, model.dfy);
%             Dpdf_pdf = (model.dfy+model.do)*(H'/model.R)*(obs-obs_mn)/(model.dfy + (obs-obs_mn)'*(model.R\(obs-obs_mn)));
%             
%             % Match a Gaussian to these
%             [y, H, R] = gaussian_match_obs(x, pdf, Dpdf_pdf);
%             
%         else
%             
%             R = model.R;
%             y = obs - obs_mn + H*x;
%             
%         end
%         
%         % Set D
%         if algo.flag_stochastic
%             if lam < 0.9
%                 dlD2 = algo.D;
%             else
%                 dlD2 = zeros(size(algo.D));
%             end
%             D = dlD2/(2*dl);
%         else
%             D = algo.D;
%         end
%         
%         % Calculate velocity
%         if ~any(isnan(P))
%             [ fA, fb ] = linear_flow( lam, m, P, y, H, R, D );
%             v = fA*x + fb;
%         else
%             v = zeros(model.ds,1);
%         end
%         
%         % Push forward
%         x = x + v*dl;
%         
%         % Stochastic bit
%         if algo.flag_stochastic
%             x = mvnrnd(x', dlD2)';
%         end
%         
%         % Store state
%         state(:,ii) = x;
%         state_evolution(:,ii,ll+1) = state(:,ii);
%         
%         % Densities
%         if ~isempty(prev_state)
%             [~, trans_prob] = feval(fh.transition, model, prev_state(:,ii), state(:,ii));
%         else
%             [~, trans_prob] = feval(fh.stateprior, model, state(:,ii));
%         end
%         [~, lhood_prob] = feval(fh.observation, model, state(:,ii), obs);
%         prob(ii) = trans_prob + lam*lhood_prob;
%         
%         % Update weight
%         inc_weight(ii) = prob(ii) - last_prob(ii) + log(det(I + dl*A));
%         weight(ii) = weight(ii) + inc_weight(ii);
%         if ~isreal(weight(ii))||isnan(weight(ii))
%             weight(ii) = -inf;
%         end
%         weight_evolution(ll+1,ii) = weight(ii);
%         
%     end
%     
%     last_prob = prob;
%     
% end

% % Weight update loop
% for ii = 1:algo.N
%     
%     % Densities
%     if ~isempty(prev_state)
%         [~, trans_prob] = feval(fh.transition, model, prev_state(:,ii), state(:,ii));
%     else
%         [~, trans_prob] = feval(fh.stateprior, model, state(:,ii));
%     end
%     [~, lhood_prob] = feval(fh.observation, model, state(:,ii), obs);
% 
%     % Weight update
%     weight(ii) = weight(ii) + lhood_prob + trans_prob - init_trans_prob(ii) + log(wt_jac(ii));
% 
% end

% % Plot particle paths (first state only)
% if display.plot_particle_paths
%     figure(1), clf, hold on
%     xlim([0 1]);
%     for ii = 1:algo.N
%         plot(lam_rng, squeeze(state_evolution(1,ii,:)));
%     end
%     figure(2), clf, hold on
%     for ii = 1:algo.N
%         plot(squeeze(state_evolution(1,ii,:)), squeeze(state_evolution(2,ii,:)));
%         plot(squeeze(state_evolution(1,ii,end)), squeeze(state_evolution(2,ii,end)), 'o');
%     end
%     figure(3), clf, hold on
%     for ii = 1:algo.N
%         plot(lam_rng, weight_evolution(:,ii));
%     end
%     drawnow;
% end

% end

function [f, fA, fb, D] = f_iter(model, lam, x, obs, A_mn, A_vr, D)

% Unpack
tau = x(1);

% Calculate value and gradient of the tau prior density
pdf = gampdf(tau-model.tau_shift, model.tau_shape, model.tau_scale);
Dpdf_pdf = (model.tau_shape-1)/(tau-model.tau_shift) - 1/model.tau_scale;

% Match a Gaussian to these
[tau_mn, tau_vr] = gaussian_match_prior(tau, pdf, Dpdf_pdf);

% Matched prior Gaussian
m = [tau_mn; A_mn];
P = diag([tau_vr, A_vr]);

% Observation mean
obs_mn = ha_h(model, x);

% Linearise observation model around the current point
H = ha_obsjacobian(model, x);

if ~isinf(model.dfy)
    
    % Calculate value and gradient of the observation density
    pdf = mvnstpdf(obs', obs_mn', model.R, model.dfy);
    Dpdf_pdf = (model.dfy+model.do)*(H'/model.R)*(obs-obs_mn)/(model.dfy + (obs-obs_mn)'*(model.R\(obs-obs_mn)));
    
    % Match a Gaussian to these
    [y, H, R] = gaussian_match_obs(x, pdf, Dpdf_pdf);
    
else
    
    R = model.R;
    y = obs - obs_mn + H*x;
    
end

% Calculate velocity
if ~any(isnan(P))
    if ~all(D(:)==0)
        D = inv(inv(P)+lam*(H'/R)*H);
    end
    [ fA, fb ] = linear_flow( lam, m, P, y, H, R, D );
    f = fA*x + fb;
else
    D = zeros(model.ds);
    f = zeros(model.ds,1);
    fb = f;
    fA = zeros(model.ds,model.ds);
end

end
