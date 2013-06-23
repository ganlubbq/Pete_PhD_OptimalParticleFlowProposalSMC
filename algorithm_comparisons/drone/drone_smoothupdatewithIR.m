function [ state, weight ] = drone_smoothupdatewithIR( display, algo, model, fh, obs, prev_state, weight)
%drone_smoothupdate Apply a smooth update for the drone model using
% intermediate resampling.

dl_start = 1E-3;

% Select resampling times
resamp_times = [0:0.1:0.2 0.5 1];
L = length(resamp_times);

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

% Step-size carry-over
last_step_sizes = dl_start*ones(1,algo.N);
step_sizes = zeros(1,algo.N);

% Loop initialisation
last_prob = init_trans_prob;

% Loop through resampling times
for ll = 1:L-1
    
%     fprintf(1,'\n%u:  ',ll);
    
    % Pseudo-time
    lam0 = resamp_times(ll);
    lam1 = resamp_times(ll+1);
    
    % Initialise state and weight arrays
    pf(ll+1).state = zeros(model.ds, algo.N);
    pf(ll+1).weight = zeros(1, algo.N);
    prob = zeros(1, algo.N);
    
    % Resampling
    if algo.flag_intermediate_resample && (calc_ESS(pf(ll).weight)<(algo.N))
        pf(ll+1).ancestor = sample_weights(pf(ll).weight, algo.N, 2);
        flag_resamp = true;
    else
        pf(ll+1).ancestor = 1:algo.N;
        flag_resamp = false;
    end
    
    % Particle loop
    for ii = 1:algo.N
        
%         fprintf(1,'%u ',ii);
        
        % Origin
        pf(ll+1).origin(ii) = pf(ll).origin(pf(ll+1).ancestor(ii));
        
        % Get states
        if ~isempty(prev_state)
            prev_x = prev_state(:,pf(ll+1).origin(ii));
        else
            prev_x = [];
        end
        x0 = pf(ll).state(:,pf(ll+1).ancestor(ii));
        
        % Update state
        [ x, ppsl_prob_upd, step_sizes(ii) ] = particleupdate( algo, model, prev_x, x0, obs, last_step_sizes(pf(ll+1).ancestor(ii)), lam0, lam1 );
        
        % Store state
        pf(ll+1).state(:,ii) = x;
        
        % Densities
        if ~isempty(prev_state)
            [~, trans_prob] = feval(fh.transition, model, prev_x, x);
        else
            [~, trans_prob] = feval(fh.stateprior, model, x);
        end
        [~, lhood_prob] = feval(fh.observation, model, x, obs);
        prob(ii) = trans_prob + lam1*lhood_prob;
        
        % Weight update
        if flag_resamp
            resamp_weight = 0;
        else
            resamp_weight = pf(ll).weight(pf(ll+1).ancestor(ii));
        end
        pf(ll+1).weight(ii) = resamp_weight + prob(ii) - last_prob(pf(ll+1).ancestor(ii)) - ppsl_prob_upd;
        
        if ~isreal(pf(ll+1).weight(ii))
            pf(ll+1).weight(ii) = -inf;
        end
        
        assert(~isnan(pf(ll+1).weight(ii)));
        
    end
    
    last_prob = prob;
    last_step_sizes = step_sizes;
    
end

state = pf(L).state;
weight = pf(L).weight;

state_evo = cat(3,pf.state);
weight_evo = cat(1,pf.weight);

% Plot
if display.plot_particle_paths
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
        plot(resamp_times, weight_traj(ii,:), 'color', [0 rand rand]);
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

function [ state, ppsl_prob, dl ] = particleupdate( algo, model, prev_state, init_state, obs, dl_start, lam_start, lam_stop )

dl_min = 1E-8;
dl_max = 0.5;
err_thresh = 0.1;
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

% Initialise loop variables
state = init_state;
ppsl_prob = 0;
dl = dl_start;
lam = lam_start;

% Sample perturbation
if algo.Dscale > 0
    zD = mvnrnd(zeros(model.ds,1)',eye(model.ds))';
else
    zD = zeros(model.ds,1);
end

% Loop
ll_count = 0;
while lam < lam_stop
    
    if ll_count > 10
        ppsl_prob = 1E10;
        break;
    end
    
    % Pseudo-time and step-size
    lam0 = lam;
    lam1 = lam + dl/(1+zD'*zD);
    if lam1 > lam_stop
        lam1 = lam_stop;
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
        xi_ppsl_prob = 0;
    else
        xi = 1;
        xi_ppsl_prob = 0;
    end
    Pxi = P / xi;
    
    % Analytical flow
    [ x, prob_ratio, drift, diffuse] = linear_flow_move( lam1, lam0, x0, m, Pxi, y, H, R, algo.Dscale, zD );
    
    % Error estimate
    H_new = drone_obsjacobian(model, x);
    y_new = obs - drone_h(model, x) + H_new*x;
    [drift_new, diffuse_new] = linear_drift( lam1, x, m, Pxi, y_new, H_new, R, algo.Dscale );
    
    deter_err_est = 0.5*(lam1-lam0)*(drift_new-drift);
    stoch_err_est = 0.5*(diffuse_new-diffuse)*zD*sqrt(lam1-lam0);
    err_est = deter_err_est + stoch_err_est;
    err_crit = err_est'*err_est;
    
    % Step size adjustment
    if (err_crit > err_thresh) || (lam1 < lam_stop)
        dl = min(dl_max, min(10*dl, dl_sf * (err_thresh/err_crit)^dl_pow * dl));
        if dl < dl_min
            warning('nlng_smoothupdatebyparticle:ErrorTolerance', 'Minimum step size reached. Local error tolerance exceeded.');
            ppsl_prob = 1E10;
            break;
        end
    end
    
    % Accept/reject step
    if err_crit < err_thresh
        
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
        
        % Update probability
        ppsl_prob = ppsl_prob - log(prob_ratio);
        
    else
        
%         disp('Error too large. Reducing step size');
        
    end
    
end





end



