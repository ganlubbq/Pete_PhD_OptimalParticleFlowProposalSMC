function [ state, ppsl_prob, ll_count ] = nlng_smoothupdatebyparticle( display, algo, model, fh, prev_state, obs )
%nlng_smoothupdatebyparticle Apply a smooth update for the nonlinear non-Gaussian
%benchmark model for a single particle (which means no intermediate
%resampling, but step size control is easier.)

ds = model.ds-1;

dl_start = 1E-3;
dl_min = 1E-7;
dl_max = 0.5;
err_thresh = 0.1;
dl_sf = 0.8;1;
dl_pow = 0.7;1;

% Prior
if isempty(prev_state)
    m = model.m1;
    P = model.P1;
else
    prev_kk = prev_state(1);
    prev_x = prev_state(2:end);
    m = nlng_f(model, prev_kk, prev_x);
    P = model.Q;
end

% Sample initial state
if ~isempty(prev_state)
    [init_state, init_trans_prob] = feval(fh.transition, model, prev_state);
else
    [init_state, init_trans_prob] = feval(fh.stateprior, model);
end

% SMoN scaling.
if ~isinf(model.dfy)
    xi = chi2rnd(model.dfy);
else
    xi = 1;
end

% Initialise evolution arrays
dl_evo = dl_start;
lam_evo = 0;
err_evo = 0;
state_evo = init_state;
xi_evo = xi;
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
    lam1 = lam + dl/(1+zD'*zD);
    if lam1 > 1
        lam1 = 1;
    end
    
    % Starting point
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
    
%     %%%%%% TESTING 1ST ORDER TS MATCHING %%%%%%%%
%     
%     R = model.R;
%     dy = obs - obs_mn;
%     dfy = model.dfy;
%     do = model.do;
%     yR = (R\dy);
%     tdist = 1 + yR'*yR/dfy;
%     
%     lhood_grad = -((dfy+do)/dfy)*H'*yR/tdist;
%     
%     [ T ] = nlng_obssecondderivtensor( model, x0 );
%     at = zeros(model.ds-1);
%     for dd = 1:model.do
%         at = at + yR(dd)*squeeze(T(:,:,dd));
%     end
%     lhood_hess = (dfy+do)*( H'*(R\H) - 2*H'*(yR*yR')*H/(dfy*tdist) - at )/(dfy*tdist);
%     
%     [hess_eigvec, hess_eigval] = eig(lhood_hess);
%     hess_eigval(hess_eigval>0) = -1;
%     lhood_hess = hess_eigvec*hess_eigval*hess_eigvec';
%     
%     %         max_eig = max(eig(prior_hess));
%     %         while ~isposdef(eye(ds) - lam*prior_hess\HRH)
%     %             prior_hess = prior_hess - 1.1 * max_eig*eye(ds);
%     %             fprintf(1,'.');
%     %         end
%     
%     P = -pinv(lhood_hess);
%     m = x0 + P*lhood_grad;
%     
%     %         assert(isposdef(P));
%     
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Analytical flow
    [ x, prob_ratio, drift, diffuse] = linear_flow_move( lam1, lam0, x0, m, P, y, H, R, algo.Dscale, zD );
    
    % Error estimate
    H_new = nlng_obsjacobian(model, x);
    y_new = obs - nlng_h(model, x) + H_new*x;
    [drift_new, diffuse_new] = linear_drift( lam1, x, m, P, y_new, H_new, R, algo.Dscale );
    %%%%%%%
%     stoch_err_vr = 0.5*(lam1-lam0)*(diffuse_new-diffuse)*(diffuse_new-diffuse)';
    deter_err_est = 0.5*(lam1-lam0)*(drift_new-drift);
%     stoch_err_est = sqrt(1-exp(-algo.Dscale*(lam1-lam0)))*(diffuse_new-diffuse)*zD/sqrt(algo.Dscale);
    stoch_err_est = 0.5*(diffuse_new-diffuse)*zD*sqrt(lam1-lam0);
    err_est = deter_err_est + stoch_err_est;
%     err_crit = deter_err_est'*deter_err_est + trace(stoch_err_vr);% + 2*sqrt(trace(2*stoch_err_vr^2));
%     err_crit = deter_err_est'*deter_err_est + stoch_err_est'*stoch_err_est;
    err_crit = err_est'*err_est;
    
    % Accept/reject step
    if err_crit < err_thresh
        
        ll_count = ll_count + 1;
        
        % Update time
        lam = lam1;
        
        % Update state
        state = [kk; x];
        
        % Sample perturbation
        if algo.Dscale > 0
            zD = mvnrnd(zeros(ds,1)',eye(ds))';
        else
            zD = zeros(ds,1);
        end
        
        % Step size adjustment
        dl = min(dl_max, min(sqrt(dl), dl_sf * (err_thresh/err_crit)^(dl_pow) * dl));
        
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
        xi_evo = [xi_evo xi];
        post_prob_evo = [post_prob_evo post_prob];
        ppsl_prob_evo = [ppsl_prob_evo ppsl_prob];
        
    else
        
        % Step size adjustment
        dl = min(dl_max, min(sqrt(dl), dl_sf * (err_thresh/err_crit)^dl_pow * dl));
        if dl < dl_min
            warning('nlng_smoothupdatebyparticle:ErrorTolerance', 'Minimum step size reached. Local error tolerance exceeded.');
            break;
        end
        
%         disp('Error too large. Reducing step size');
        
    end
    
end

% Plotting
if display.plot_particle_paths
    
    % 2D state trajectories
    figure(display.h_ppp(1));
    plot(state_evo(2,:), state_evo(3,:), ':');
    plot(init_state(2), init_state(3), 'o');
    plot(state(2), state(3), 'xr', 'markersize', 8);
    
    % 1D state trajectories
    figure(display.h_ppp(2));
    plot(lam_evo, state_evo(3,:));
    
    % Step size
    figure(display.h_ppp(3));
    plot(lam_evo, dl_evo);
    
    % Error estimates
    figure(display.h_ppp(4));
    plot(lam_evo, err_evo);
    
    % Probability estimate
    figure(display.h_ppp(5));
    plot(lam_evo, post_prob_evo-ppsl_prob_evo);
    
    % Mixing variable
    figure(display.h_ppp(6));
    plot(lam_evo, xi_evo);
    
end

end

