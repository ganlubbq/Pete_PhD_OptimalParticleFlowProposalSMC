function [ state, ppsl_prob ] = sineha_smoothupdatebyparticle( display, algo, model, fh, prev_state, obs )
%ha_smoothupdate Apply a smooth update for the heartbeat alignment model
%for a single particle (which means no intermediate resampling, but step
%size control is easier.)

ds = model.ds;

dl_start = 1E-5;
dl_min = 1E-6;
dl_max = 0.5;
err_thresh = 0.001;
dl_sf = 0.8;
dl_pow = 0.7;

% Prior
if isempty(prev_state)
    prior_mn = [model.A_shape*model.A_scale+model.A_shift;
                model.T1_mn;
                model.tau_shape*model.tau_scale;
                model.omega1_mn;
                model.phi1_mn;
                model.B1_mn];
else
    prior_mn = [model.A_shape*model.A_scale+model.A_shift;
                prev_state(2);
                model.tau_shape*model.tau_scale;
                prev_state(4:6)];
end
prior_vr = diag([model.A_shape*model.A_scale^2;
                 (exp(model.T_vol)-1)*exp(2*prior_mn(2)+model.T_vol);
                 model.tau_shape*model.tau_scale^2;
                 model.omega_vr;
                 model.phi_vr;
                 model.B_vr]);

% Sample initial state
if ~isempty(prev_state)
    [init_state, init_trans_prob] = feval(fh.transition, model, prev_state);
else
    [init_state, init_trans_prob] = feval(fh.stateprior, model);
end

% Initialise evolution arrays
dl_evo = dl_start;
lam_evo = 0;
err_evo = 0;
state_evo = init_state;
post_prob_evo = init_trans_prob;
ppsl_prob_evo = init_trans_prob;

% Initialise loop variables
state = init_state;
post_prob = init_trans_prob;
ppsl_prob = init_trans_prob;
dl = dl_start;
lam = 0;

% Sample perturbation
if algo.Dscale > 0
    zD = mvnrnd(zeros(ds,1)',eye(ds))';
else
    zD = zeros(ds,1);
end

% Loop
ll_count = 0;
while lam < 1
    
    if ll_count > 50
        ppsl_prob = 1E10;
        break;
    end
    
    % Pseudo-time and step-size
    lam0 = lam;
    lam1 = lam + dl;
    if lam1 > 1
        lam1 = 1;
    end
    
    % Starting point
    x0 = state;
    
    % Observation mean
    obs_mn = sineha_h(model, x0);
    
    % Linearise observation model around the current point
    H = sineha_obsjacobian(model, x0);
    y = obs - obs_mn + H*x0;
    R = model.R;
    
    % Approximate prior
    [P, m] = diff_prior(model, x0, prior_mn, prior_vr);
    
    % Analytical flow
    [ x, prob_ratio, drift, diffuse] = linear_flow_move( lam1, lam0, x0, m, P, y, H, R, algo.Dscale, zD );
    
    % Error estimate
    [P_new, m_new] = diff_prior(model, x, prior_mn, prior_vr);
    H_new = sineha_obsjacobian(model, x);
    y_new = obs - sineha_h(model, x) + H_new*x;
    [drift_new, diffuse_new] = linear_drift( lam1, x, m_new, P_new, y_new, H_new, R, algo.Dscale );

    deter_err_est = 0.5*(lam1-lam0)*(drift_new-drift);
    stoch_err_est = 0.5*(diffuse_new-diffuse)*zD*sqrt(lam1-lam0);
    err_est = deter_err_est + stoch_err_est;
    err_crit = err_est'*err_est;
    
    % Step size adjustment
    dl = min(dl_max, min(10*dl, dl_sf * (err_thresh/err_crit)^dl_pow * dl));
    if dl < dl_min
%         dl = dl_min;
%         warning('ha_smoothupdatebyparticle:ErrorTolerance', 'Minimum step size reached. Local error tolerance exceeded.');
        break;
    end
    
    % Accept/reject step
    if (err_crit < err_thresh) || (dl == dl_min)
        
        ll_count = ll_count + 1;
        
        % Update time
        lam = lam1;
        
        % Update state
        state = x;
        
        % Sample perturbation
        if algo.Dscale > 0
            zD = mvnrnd(zeros(ds,1)',eye(ds))';
        else
            zD = zeros(ds,1);
        end
        
        % Densities
        if ~isempty(prev_state)
            [~, trans_prob] = feval(fh.transition, model, prev_state, state);
        else
            [~, trans_prob] = feval(fh.stateprior, model, state);
        end
        [~, lhood_prob] = feval(fh.observation, model, state, obs);
        
        % Update probabilities
        post_prob = trans_prob + lam*lhood_prob;
        ppsl_prob = ppsl_prob - log(prob_ratio);
        
        % Update evolution
        dl_evo = [dl_evo dl];
        lam_evo = [lam_evo lam];
        err_evo = [err_evo err_crit];
        state_evo = [state_evo state];
        post_prob_evo = [post_prob_evo post_prob];
        ppsl_prob_evo = [ppsl_prob_evo ppsl_prob];
        
        %%%%%%%%%%%%%%%%%%%%%%%
        if post_prob-ppsl_prob < - 100
            ppsl_prob = 1E10;
            break;
        end
        if (~isreal(ppsl_prob-post_prob))||(~isreal(state))
            state = real(state);
            ppsl_prob = 1E10;
            break
        end
        %%%%%%%%%%%%%%%%%%%%%%%
        
    else
        
%         disp('Error too large. Reducing step size');
        
    end
    
end

% Plotting
if display.plot_particle_paths
    
    % 2D state trajectories
    figure(display.h_ppp(1));
    plot(state_evo(2,:), state_evo(3,:));
    plot(init_state(2), init_state(3), 'o');
    plot(state(2), state(3), 'xr');
    
    % 1D state trajectories
    figure(display.h_ppp(2));
    plot(lam_evo, state_evo(2,:));
    figure(display.h_ppp(3));
    plot(lam_evo, state_evo(3,:));
    
    % Step size
    figure(display.h_ppp(4));
    plot(lam_evo, dl_evo);
    
    % Error estimates
    figure(display.h_ppp(5));
    plot(lam_evo, err_evo);
    
    % Probability estimate
    figure(display.h_ppp(6));
    plot(lam_evo, post_prob_evo-ppsl_prob_evo);
    
end

end

function [P, m] = diff_prior(model, x, prior_mn, prior_vr)

% P = prior_vr;
% m = prior_mn;

% Unpack state
A = x(1);
T = x(2);
tau = x(3);
omega = x(4);
phi = x(5);
B = x(6);

% tau_scale = prior_mn(3)/model.tau_shape;
Tminusmean = log(T)-log(prior_mn(2))+0.5*model.T_vol;

% Prior matching
grad_prior = [(model.A_shape-1)/(A-model.A_shift)-1/model.A_scale;
%     -1/T-Tminusmean/(T*model.T_vol)-(model.tau_shape-1)/(tau-T)+1/model.tau_scale;
%     (model.tau_shape-1)/(tau-T)-1/model.tau_scale;
    -1/T-Tminusmean/(T*model.T_vol);
    (model.tau_shape-1)/(tau)-1/model.tau_scale;
    -(omega-prior_mn(4))/model.omega_vr;
    -(phi-prior_mn(5))/model.phi_vr;
    -(B-prior_mn(6))/model.B_vr];
hess_prior = diag([-(model.A_shape-1)/((A-model.A_shift)^2);
    %         ((Tminusmean - 1)/model.T_vol+1)/T^2 - (model.tau_shape-1)/(tau-T)^2;
    %         -(model.tau_shape-1)/(tau-T)^2;
        ((Tminusmean - 1)/model.T_vol+1)/T^2;
        -(model.tau_shape-1)/(tau)^2;
        -1/model.omega_vr;
        -1/model.phi_vr;
        -1/model.B_vr]);
%     off_diagional = (model.tau_shape-1)/(tau-T)^2;
%     hess_prior(2,3) = off_diagional;
%     hess_prior(3,2) = off_diagional;

% % Adjustments
% P_sub = -inv(hess_prior(2:3,2:3));
% [V, D] = eig(P_sub);
% limits = [(exp(model.T_vol)-1)*exp(2)*prior_mn(2),  model.tau_shape*model.tau_scale^2];
% max_vr = limits*abs(V);
% D = min(D, diag(max_vr));
% P_sub = V*D*V';
% hess_prior(2:3,2:3) = -inv(P_sub);

P_sub = -1/hess_prior(1,1);
max_vr = model.A_shape*model.A_scale^2;
P_sub(P_sub>max_vr) = max_vr;
hess_prior(1,1) = -1/P_sub;

P_sub = -1/hess_prior(2,2);
max_vr = (exp(model.T_vol)-1)*exp(2)*prior_mn(2);
P_sub(P_sub>max_vr) = max_vr;
P_sub(P_sub<0) = max_vr;
hess_prior(2,2) = -1/P_sub;

P_sub = -1/hess_prior(3,3);
max_vr = model.tau_shape*model.tau_scale^2;
P_sub(P_sub>max_vr) = max_vr;
hess_prior(3,3) = -1/P_sub;

P = -inv(hess_prior);
m = x + P*grad_prior;

end

