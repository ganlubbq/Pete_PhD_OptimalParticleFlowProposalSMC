function [ state, weight, state_evolution ] = ha_smoothupdate( display, algo, model, fh, obs, prev_state, weight)
%ha_smoothupdate Apply a smooth update for the heartbeat alignment model.

% Sample prior
init_trans_prob = zeros(1, algo.N);
state = zeros(model.ds, algo.N);
for ii = 1:algo.N
    if ~isempty(prev_state)
        [state(:,ii), init_trans_prob(ii)] = feval(fh.transition, model, prev_state(:,ii));
    else
        [state(:,ii), init_trans_prob(ii)] = feval(fh.stateprior, model);
    end
end

% Set up integration schedule
% num_steps = 200;ratio = 1.03;
num_steps = 100;ratio = 1.07;
scale_fact = (1-ratio)/(ratio*(1-ratio^num_steps));
lam_rng = cumsum([0 scale_fact*ratio.^(1:num_steps)]);
% lam_rng = [1E-5:1E-5:1E-4 2E-4:1E-4:1E-3 2E-3:1E-3:1E-2 2E-2:1E-2:1E-1 2E-1:1E-1:9E-1 9.1E-1:1E-2:1];
L = length(lam_rng);

% State evolution array (in case we want to plot the trajectories)
state_evolution = zeros(model.ds, algo.N, L);
state_evolution(:,:,1) = state;
weight_evolution = zeros(L,algo.N);
weight_evolution(1,:) = weight;

% Weight arrays
inc_weight = zeros(1, algo.N);
prob = zeros(1, algo.N);
last_prob = init_trans_prob;

% Prior
if isempty(prev_state)
    A_mn_arr = repmat(model.A1_mn, 1, algo.N);
    A_vr = model.A1_vr;
else
    A_mn_arr = prev_state(2,:);
    A_vr = model.A_vr;
end

% Other variables
dsc = model.ds - 1;
I = eye(dsc);

if display.plot_particle_paths
    figure(1); clf; hold on;
end

for ii = 1:algo.N
    
    x = state(:,ii);
    A_mn = A_mn_arr(1,ii);
    
    lam_rng = [0 1];
    [lam, x] = ode45(@(lam_in, x_in) v_iter(model, lam_in, x_in, obs, A_mn, A_vr), lam_rng, x);
    state(:,ii) = x(end,:)';
    
    % Densities
    if ~isempty(prev_state)
        [~, trans_prob] = feval(fh.transition, model, prev_state(:,ii), state(:,ii));
    else
        [~, trans_prob] = feval(fh.stateprior, model, state(:,ii));
    end
    [~, lhood_prob] = feval(fh.observation, model, state(:,ii), obs);
    
    % Update weight
    weight(ii) = weight(ii) + trans_prob + lhood_prob - init_trans_prob(ii);
    if ~isreal(weight(ii))||isnan(weight(ii))
        weight(ii) = -inf;
    end
    
    if display.plot_particle_paths
        plot(x(:,1), x(:,2));
        plot(x(end,1), x(end,2), 'o');
    end
    
end

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

end

function v = v_iter(model, lam, x, obs, A_mn, A_vr)

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

D = zeros(model.ds);

% Calculate velocity
if ~any(isnan(P))
    [ fA, fb ] = linear_flow( lam, m, P, y, H, R, D );
    v = fA*x + fb;
else
    v = zeros(model.ds,1);
end

end
