function [ state, weight, state_evolution ] = nlng_smoothupdate( display, algo, model, fh, obs, prev_state, weight)
%nlng_smoothupdate Apply a smooth update for the nonlinear non-Gaussian
%benchmark model.

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
ratio = 1.05;
num_steps = 100;
scale_fact = (1-ratio)/(ratio*(1-ratio^num_steps));
lam_rng = cumsum([0 scale_fact*ratio.^(1:num_steps)]);
% lam_rng = 0:0.01:1;
L = length(lam_rng);

% State evolution array (in case we want to plot the trajectories)
state_evolution = zeros(model.ds, algo.N, L);
state_evolution(:,:,1) = state;
weight_evolution = zeros(L,algo.N);
weight_evolution(1,:) = weight;

% Weight arrays
prob = zeros(1, algo.N);
last_prob = init_trans_prob;

% Prior density
if isempty(prev_state)
    prior_mn = repmat(model.m1, 1, algo.N);
    P = model.P1;
else
    prev_kk = prev_state(1);
    prev_x = prev_state(2:end,:);
    P = model.Q;
    prior_mn = zeros(size(prev_x));
    for ii = 1:algo.N
        prior_mn(:,ii) = nlng_f(model, prev_kk, prev_x(:,ii));
    end
end

% Other variables
dsc = model.ds - 1;
I = eye(dsc);

% Pseudo-time loop
for ll = 1:L-1
    
    % Resampling
    
    % Pseudo-time
    lam0 = lam_rng(ll);
    lam = lam_rng(ll+1);
    
    % Particle loop
    for ii = 1:algo.N
        
        % Get state
        x0 = state(2:end,ii);
        m = prior_mn(:,ii);
        
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
            [y, H, R] = gaussian_match_obs(x0, p, Dp_p);
%             [y, ~] = gaussian_match_obs(x0, Dp, [], H, R);
%             y = obs;
            
        else
            
            R = model.R;
            y = obs - obs_mn + H*x0;
            
        end
        
        % Analytical flow
        [ x, wt_jac, prob_ratio] = linear_flow_move( lam, lam0, x0, m, P, y, H, R, algo.Dscale );
        
        % Store state
        state(2:end,ii) = x;
        state_evolution(:,ii,ll+1) = state(:,ii);
        
        % Densities
        if ~isempty(prev_state)
            [~, trans_prob] = feval(fh.transition, model, prev_state(:,ii), state(:,ii));
        else
            [~, trans_prob] = feval(fh.stateprior, model, state(:,ii));
        end
        [~, lhood_prob] = feval(fh.observation, model, state(:,ii), obs);
        prob(ii) = trans_prob + lam*lhood_prob;
        
        % Weight update
        if algo.Dscale == 0
            weight(ii) = weight(ii) + prob(ii) - last_prob(ii) + log(wt_jac);
        else
            weight(ii) = weight(ii) + prob(ii) - last_prob(ii) + prob_ratio;%log(wt_jac);%
        end
        weight_evolution(ll+1,ii) = weight(ii);
        
        if ~isreal(weight(ii))
            weight(ii) = -inf;
        end
        
        %%% REMOVE THIS %%%
        if isnan(weight(ii))
            weight(ii) = -inf;
        end
        %%%%%%%%%%%%%%%%%%%
        
        assert(~isnan(weight(ii)));
        
    end
    
    last_prob = prob;
    test(ll) = calc_ESS(weight);
    
end

% Plot particle paths (first state only)
if display.plot_particle_paths
    figure(1), clf, hold on
    xlim([0 1]);
    for ii = 1:algo.N
        plot(lam_rng, squeeze(state_evolution(2,ii,:)));
    end
    figure(2), clf, hold on
    for ii = 1:algo.N
        plot(squeeze(state_evolution(2,ii,:)), squeeze(state_evolution(3,ii,:)));
        plot(squeeze(state_evolution(2,ii,end)), squeeze(state_evolution(3,ii,end)), 'o');
    end
    figure(3), clf, hold on
    for ii = 1:algo.N
        plot(lam_rng, weight_evolution(:,ii));
    end
    drawnow;
end

end

function v = v_iter(model, lam, x, obs, m, P)

H = nlng_obsjacobian(model, x);
obs_mn = nlng_h(model, x);

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

[A, b] = linear_flow(lam, m, P, y, H, R);
v = A*x+b;

end
