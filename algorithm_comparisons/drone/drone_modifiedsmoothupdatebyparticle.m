function [ state, ppsl_prob, ll_count, loglhood, expectloglhoodest ] = drone_modifiedsmoothupdatebyparticle( display, algo, model, fh, prev_state, obs )
%drone_smoothupdatebyparticle Apply a smooth update for the drone model
%for a single particle (which means no intermediate resampling, but step
%size control is easier.)

assert(isinf(model.dfx));

dl_start = 1E-3;
dl_min = 1E-8;
dl_max = 0.5;
err_thresh = 0.1;
dl_sf = 0.8;
dl_pow = 0.7;

if ~isinf(model.dfx)
    dl_max = 0.05;
end

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

% Sample perturbation
if algo.Dscale > 0
    zD = mvnrnd(zeros(model.ds,1)',eye(model.ds))';
else
    zD = zeros(model.ds,1);
end

mu = m;
Sigma = P;

% Loop
ll_count = 0;
while lam < 1
    
    if ll_count > 5000
        ppsl_prob = 1E10;
        break;
    end
    
    % Pseudo-time and step-size
    lam0 = lam;
    lam1 = lam + dl;%/(1+zD'*zD);
    if lam1 > 1
        lam1 = 1;
    end
    
    % Starting point
    x0 = state;
    
    % Observation mean
    obs_mn = drone_h(model, x0);
    
    % Linearise observation model around the current point
    H = drone_obsjacobian(model, x0);
    ymhx = obs - obs_mn;
    R = model.R;
    
    % Resolve bearing ambiguity
    if ymhx(1) > pi
        ymhx(1) = ymhx(1) - 2*pi;
    elseif ymhx(1) < -pi
        ymhx(1) = ymhx(1) + 2*pi;
    end
    
    % Analytical flow
    [ x, mu, Sigma, prob_ratio, drift, diffuse] = modified_linear_flow_move( lam1, lam0, x0, mu, Sigma, ymhx, H, R, algo.Dscale, zD );
    
    % Error estimate
    H_new = drone_obsjacobian(model, x);
    ymhx_new = obs - drone_h(model, x);
    [drift_new, diffuse_new] = modified_linear_drift( lam1, x, mu, Sigma, ymhx_new, H_new, R, algo.Dscale );
    
%     deter_err_est = 0.5*(lam1-lam0)*(drift_new-drift);
%     stoch_err_est = 0.5*(diffuse_new-diffuse)*zD*sqrt(lam1-lam0);
% %     stoch_err_est = 0.5*(diffuse_new-diffuse)*zD*(lam1-lam0);
% %     stoch_err_est = 0.5*(diffuse_new-diffuse)*zD*sqrt(1-exp(-algo.Dscale*(lam1-lam0)));
%     err_est = deter_err_est + stoch_err_est;
%     err_crit = sqrt(err_est'*err_est);

    err_est = 0.5*(lam1-lam0)*( (drift_new-drift) + (diffuse_new-diffuse)*zD );
    err_crit = sqrt(err_est'*err_est);
    
%     ldiff = (lam1-lam0);
%     driftdiff = (drift_new-drift);
%     diffusediff = (diffuse_new-diffuse);
%     err_crit = 0.5*sqrt( ldiff^2*(driftdiff'*driftdiff) + ldiff*trace(diffusediff*diffusediff') );
    
    % Step size adjustment
    dl = min(dl_max, min(10*dl, dl_sf * (err_thresh/err_crit)^dl_pow * dl));
    if dl < dl_min
%         dl = dl_min;
%         warning('nlng_smoothupdatebyparticle:ErrorTolerance', 'Minimum step size reached. Local error tolerance exceeded.');
        ppsl_prob = 1E10;
        break;
    end
    
    % Accept/reject step
    if 1%(algo.Dscale > 0) || (err_crit < err_thresh) % We can only reject steps if we're deterministic
        
        ll_count = ll_count + 1;
        
        % Update time
        lam = lam1;
        
        % Update state
        state = x;
        
        % Sample perturbation
        if algo.Dscale > 0
            zD = mvnrnd(zeros(model.ds,1)',eye(model.ds))';
        else
            zD = zeros(model.ds,1);
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
        
    else
        
%         disp('Error too large. Reducing step size');
        
    end
    
end

R =model.R;
H = drone_obsjacobian(model, state);
h_x = drone_h(model, state);
x = state;
y = obs-h_x;
Sigma = inv(inv(P)+H'*(R\H));
mu = Sigma*(H'*(R\y)+P\(m-x));
loglhood = loggausspdf(obs, h_x, R);
expectloglhoodest = loggausspdf(y, H*mu, R) - 0.5*trace(Sigma*H'*(R\H));

% Plotting
if display.plot_particle_paths
    
    % 2D state trajectories
    figure(display.h_ppp(1));
    plot(state_evo(1,:), state_evo(2,:), ':');
    plot(init_state(1), init_state(2), 'o');
    plot(state(1), state(2), 'xr', 'markersize', 8);
    
    % 2D state trajectories
    figure(display.h_ppp(2));
    plot(state_evo(1,:), state_evo(3,:), ':');
    plot(init_state(1), init_state(3), 'o');
    plot(state(1), state(3), 'xr', 'markersize', 8);
    
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

