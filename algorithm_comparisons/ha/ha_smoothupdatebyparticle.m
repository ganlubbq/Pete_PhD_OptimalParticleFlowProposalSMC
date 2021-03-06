function [ state, ppsl_prob ] = ha_smoothupdatebyparticle( display, algo, model, fh, prev_state, obs )
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

% Amplitude prior
if isempty(prev_state)
    A_mn = model.A1_mn;
    A_vr = model.A1_vr;
else
    A_mn = prev_state(2);
    A_vr = model.A_vr;
end

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
while lam < 1
    
    % Pseudo-time and step-size
    lam0 = lam;
    lam1 = lam + dl;
    if lam1 > 1
        lam1 = 1;
    end
    
    % Starting point
    x0 = state;
    
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
    
    % Features of prior density
    tau = x0(1);
    prior_grad = (model.tau_shape-1)./(tau-model.tau_shift) - 1/model.tau_scale;
    
    % GRADIENT-HESSIAN MATCHING
    prior_hess = -(model.tau_shape-1)./(tau-model.tau_shift)^2;
   tau_vr = min(-1/prior_hess,0.02);
    tau_mn = tau + tau_vr*prior_grad;
        
%     % VALUE-GRADIENT MATCHING
%     prior_value = gampdf(tau-model.tau_shift, model.tau_shape, model.tau_scale);
%     if (tau > model.tau_shift) && (prior_value > 0)
%         [tau_mn, tau_vr] = gaussian_match_prior(tau, prior_value, prior_grad);
%     else
%         tau_mn = model.tau_shape * model.tau_scale;
%         tau_vr = model.tau_shape * model.tau_scale^2;
%     end

%     % MOMENT MATCHING
%     tau_mn = model.tau_shape * model.tau_scale;
%     tau_vr = model.tau_shape * model.tau_scale^2;

%     % MEAN-GRADIENT MATCHING
%     tau_mn = model.tau_shape * model.tau_scale;
%     tau_vr = (tau_mn-tau)/prior_grad;
%     if tau_vr < 0
%         tau_vr = model.tau_shape * model.tau_scale^2;
%     end

%     % IGNORING
%     tau_vr = 1E3;
%     tau_mn = 0;
        
    % Matched prior Gaussian
    m = [tau_mn; A_mn];
    P = diag([tau_vr, A_vr]);
    
    % Analytical flow
    [ x, prob_ratio, drift, diffuse] = linear_flow_move( lam1, lam0, x0, m, P, y, H, R, algo.Dscale, zD );
    
    % Error estimate
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Features of prior density
    tau = x(1);
    prior_grad = (model.tau_shape-1)./(tau-model.tau_shift) - 1/model.tau_scale;
    
    % GRADIENT-HESSIAN MATCHING
    prior_hess = -(model.tau_shape-1)./(tau-model.tau_shift)^2;    
    tau_vr = min(-1/prior_hess,0.02);
    tau_mn = tau + tau_vr*prior_grad;
        
%     % VALUE-GRADIENT MATCHING
%     prior_value = gampdf(tau-model.tau_shift, model.tau_shape, model.tau_scale);
%     if (tau > model.tau_shift) && (prior_value > 0)
%         [tau_mn, tau_vr] = gaussian_match_prior(tau, prior_value, prior_grad);
%     else
%         tau_mn = model.tau_shape * model.tau_scale;
%         tau_vr = model.tau_shape * model.tau_scale^2;
%     end

%     % MOMENT MATCHING
%     tau_mn = model.tau_shape * model.tau_scale;
%     tau_vr = model.tau_shape * model.tau_scale^2;

%     % MEAN-GRADIENT MATCHING
%     tau_mn = model.tau_shape * model.tau_scale;
%     tau_vr = (tau_mn-tau)/prior_grad;
%     if tau_vr < 0
%         tau_vr = model.tau_shape * model.tau_scale^2;
%     end

%     % IGNORING
%     tau_vr = 1E3;
%     tau_mn = 0;

    % Matched prior Gaussian
    m_new = [tau_mn; A_mn];
    P_new = diag([tau_vr, A_vr]);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    H_new = ha_obsjacobian(model, x);
    y_new = obs - ha_h(model, x) + H_new*x;
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
            break;
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
    plot(state_evo(1,:), state_evo(2,:));
    plot(init_state(1), init_state(2), 'o');
    plot(state(1), state(2), 'xr');
    
    % 1D state trajectories
    figure(display.h_ppp(2));
    plot(lam_evo, state_evo(1,:));
    figure(display.h_ppp(3));
    plot(lam_evo, state_evo(2,:));
    
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

