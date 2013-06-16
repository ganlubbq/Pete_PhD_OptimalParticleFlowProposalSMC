function [ state, ppsl_prob ] = drone_smoothupdatebyparticle( display, algo, model, fh, prev_state, obs )
%drone_smoothupdatebyparticle Apply a smooth update for the drone model
%for a single particle (which means no intermediate resampling, but step
%size control is easier.)

dl_start = 1E-3;
dl_min = 1E-8;
dl_max = 0.5;
err_thresh = 1;
dl_sf = 0.8;
dl_pow = 0.7;

% Prior
if isempty(prev_state)
    m = model.m1;
    P = model.P1;
else
    m = model.A*prev_state;
    P = model.Q;
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

% Loop
ll_count = 0;
while lam < 1
    
    if ll_count > 100
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
    obs_mn = drone_h(model, x0);
    
    % Linearise observation model around the current point
    H = drone_obsjacobian(model, x0);
    y = obs - obs_mn + H*x0;
    R = model.R;
    
    % Resolve bearing ambiguity
    if y(1) > pi
        y(1) = y(1) - 2*pi;
    elseif y(1) < -pi
        y(1) = y(1) + 2*pi;
    end
    
    % SMoN scaling.
    if ~isinf(model.dfx)
        xi = chi2rnd(model.dfx);
    else
        xi = 1;
    end
    Pxi = P / xi;
    
    % Analytical flow
    [ x, wt_jac, prob_ratio, drift] = linear_flow_move( lam1, lam0, x0, m, Pxi, y, H, R, algo.Dscale );
    
    % Error estimate
    H_new = drone_obsjacobian(model, x);
    y_new = obs - drone_h(model, x) + H_new*x;
    drift_new = linear_drift( lam1, x, m, Pxi, y_new, H_new, R, algo.Dscale );
    err_est = 0.5*dl*( drift_new - drift );
    
    % Step size adjustment
    err_crit = norm(err_est, 2);
    dl = min(dl_max, min(10*dl, dl_sf * (err_thresh/err_crit)^dl_pow * dl));
    if dl < dl_min
%         dl = dl_min;
%         warning('nlng_smoothupdatebyparticle:ErrorTolerance', 'Minimum step size reached. Local error tolerance exceeded.');
        break;
    end
    
    % Accept/reject step
    if err_crit < err_thresh
        
        ll_count = ll_count + 1;
        
        % Update time
        lam = lam1;
        
        % Update state
        state = x;
        
        % Densities
        if ~isempty(prev_state)
            [~, trans_prob] = feval(fh.transition, model, prev_state, state);
        else
            [~, trans_prob] = feval(fh.stateprior, model, state);
        end
        [~, lhood_prob] = feval(fh.observation, model, state, obs);
        
        % Update probabilities
        post_prob = trans_prob + lam*lhood_prob;
        if algo.Dscale == 0
            ppsl_prob = ppsl_prob - log(wt_jac);
        else
            ppsl_prob = ppsl_prob - log(prob_ratio);
        end
        
        % Update evolution
        dl_evo = [dl_evo dl];
        lam_evo = [lam_evo lam];
        err_evo = [err_evo err_crit];
        state_evo = [state_evo state];
        post_prob_evo = [post_prob_evo post_prob];
        ppsl_prob_evo = [ppsl_prob_evo ppsl_prob];
        
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
    
    % Step size
    figure(display.h_ppp(3));
    plot(lam_evo, dl_evo);
    
    % Error estimates
    figure(display.h_ppp(4));
    plot(lam_evo, err_evo);
    
    % Probability estimate
    figure(display.h_ppp(5));
    plot(lam_evo, post_prob_evo-ppsl_prob_evo);
    
end

end

