function [ state, weight ] = drone_smoothupdatewithRM( display, algo, model, fh, obs, prev_state, weight)
%drone_smoothupdate Apply a smooth update for the drone model using
% intermediate resampling.

if display.plot_particle_paths
    figure(10);
    clf;
    hold on
end

dl_start = 1E-3;

% Initialise particle filter structure
pf = repmat(struct('state', [], 'ancestor', [], 'weight', []), 1, 3);

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

% Pseudo-time
lam0 = 0;
lam1 = 1;

% Initialise state and weight arrays
pf(2).state = zeros(model.ds, algo.N);
pf(2).weight = zeros(1, algo.N);
prob = zeros(1, algo.N);
ppsl_prob_upd = zeros(1, algo.N);

% Ancestors
pf(2).ancestor = 1:algo.N;

% Particle loop
for ii = 1:algo.N
    
    %         fprintf(1,'%u ',ii);
    
    % Origin
    pf(2).origin(ii) = pf(1).origin(pf(2).ancestor(ii));
    
    % Get states
    if ~isempty(prev_state)
        prev_x = prev_state(:,pf(2).origin(ii));
    else
        prev_x = [];
    end
    x0 = pf(1).state(:,pf(2).ancestor(ii));
    
    % Update state
    [ x, ppsl_prob_upd(ii) ] = particleupdate( display, algo, model, prev_x, x0, obs, dl_start, lam0, lam1 );
    
    % Store state
    pf(2).state(:,ii) = x;
    
    % Densities
    if ~isempty(prev_state)
        [~, trans_prob] = feval(fh.transition, model, prev_x, x);
    else
        [~, trans_prob] = feval(fh.stateprior, model, x);
    end
    [~, lhood_prob] = feval(fh.observation, model, x, obs);
    prob(ii) = trans_prob + lam1*lhood_prob;
    
    % Weight update
    pf(2).weight(ii) = pf(1).weight(pf(2).ancestor(ii)) + prob(ii) - (init_trans_prob(pf(2).ancestor(ii)) + ppsl_prob_upd(ii));
    
    if ~isreal(pf(2).weight(ii))
        pf(2).weight(ii) = -inf;
    end
    
    assert(~isnan(pf(2).weight(ii)));
    
end

if display.plot_particle_paths
    figure(10);
    clf
    hold on
end

calc_ESS(pf(2).weight)

% pf(3) = pf(2);
% Final selection step
pf(3).state = zeros(model.ds, algo.N);
pf(3).weight = zeros(1, algo.N);
pf(3).ancestor = sample_weights(pf(2).weight, algo.N, 2);
last_anc = 0;
accept_count = 0;
for ii = 1:algo.N
    anc = pf(3).ancestor(ii);
    pf(3).state(:,ii) = pf(2).state(:,anc);
    pf(3).weight(ii) = 0;
    pf(3).origin(ii) = pf(2).origin(anc);
    
    % MCMC move
    %if anc == last_anc
        
        anc_anc = pf(2).ancestor(anc);
        if ~isempty(prev_state)
            prev_x = prev_state(:,pf(2).origin(anc));
        else
            prev_x = [];
        end
        x0 = pf(1).state(:,anc_anc);
        [ x, new_ppsl_prob_upd ] = particleupdate( display, algo, model, prev_x, x0, obs, dl_start, lam0, lam1 );
        if ~isempty(prev_state)
            [~, trans_prob] = feval(fh.transition, model, prev_x, x);
        else
            [~, trans_prob] = feval(fh.stateprior, model, x);
        end
        [~, lhood_prob] = feval(fh.observation, model, x, obs);
        new_prob = trans_prob + lam1*lhood_prob;
        
        if log(rand) < ((new_prob-prob(anc))-(new_ppsl_prob_upd-ppsl_prob_upd(anc)))
            
            accept_count = accept_count + 1;
            pf(3).state(:,ii) = x;
            
        end
        
    %end
    
    last_anc = anc;
end

accept_count

state = pf(3).state;
weight = pf(3).weight;

% Plot
L = 2;
if display.plot_particle_paths
    weight_traj = zeros(algo.N,L+1);
    state_traj = zeros(2,algo.N,L+1);
    figure(1), clf, hold on
    xlim([0 1]);
    for ii = 1:algo.N
        weight_traj(ii,L+1) = weight(ii);
        idx = ii;
        for ll = L:-1:1
            idx = pf(ll+1).ancestor(idx);
            weight_traj(ii,ll) = pf(ll).weight(idx);
        end
        plot([0 1 1], weight_traj(ii,:), 'color', [0 rand rand]);
    end
    figure(2), clf, hold on
    for ii = 1:algo.N
        state_traj(:,ii,L+1) = state(1:2,ii);
        idx = ii;
        for ll = L:-1:1
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

function [ state, ppsl_prob, dl ] = particleupdate( display, algo, model, prev_state, init_state, obs, dl_start, lam_start, lam_stop )

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

% SMoN scaling.
if ~isinf(model.dfx)
    xi = chi2rnd(model.dfx);
else
    xi = 1;
end
if ~isempty(prev_state)
    Pxi = P / xi;
else
    Pxi = P;
end

state_evo = init_state;

% Loop
ll_count = 0;
while lam < lam_stop
    
    if ll_count > 50
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
        
%         dec = 0.1; dfx = model.dfx;
%         xi_mhppsl = (1-dec)*xi+gamrnd(0.5*dec*dfx/(1-dec),2*(1-dec));
%         new_prob = log(chi2pdf(xi_mhppsl, dfx));
%         old_prob = log(chi2pdf(xi, dfx));
%         new_ppsl = log(gampdf(xi_mhppsl-(1-dec)*xi, 0.5*dec*dfx/(1-dec), 2*(1-dec)));
%         old_ppsl = log(gampdf(xi-(1-dec)*xi_mhppsl, 0.5*dec*dfx/(1-dec), 2*(1-dec)));
%         if log(rand)<((new_prob-old_prob)-(new_ppsl-old_ppsl))
%             xi = xi_mhppsl;
%         end
        
    else
        xi = 1;
    end
    if ~isempty(prev_state)
        Pxi = P / xi;
    else
        Pxi = P;
    end
    
    % Analytical flow
    [ x, prob_ratio, drift, diffuse] = linear_flow_move( lam1, lam0, x0, m, Pxi, y, H, R, algo.Dscale, zD );
    
    state_evo = [state_evo x];
    
    % Error estimate
    H_new = drone_obsjacobian(model, x);
    y_new = obs - drone_h(model, x) + H_new*x;
    [drift_new, diffuse_new] = linear_drift( lam1, x, m, Pxi, y_new, H_new, R, algo.Dscale );
    
%     deter_err_est = 0.5*(lam1-lam0)*(drift_new-drift);
%     stoch_err_est = 0.5*(diffuse_new-diffuse)*zD*sqrt(lam1-lam0);
%     err_est = deter_err_est + stoch_err_est;
%     err_crit = err_est'*err_est;
    err_est = 0.5*(lam1-lam0)*( (drift_new-drift) + (diffuse_new-diffuse)*zD );
    err_crit = sqrt(err_est'*err_est);
    
    % Step size adjustment
    if (err_crit > err_thresh) || (lam1 < lam_stop)
        dl = min(dl_max, min(10*dl, dl_sf * (err_thresh/err_crit)^dl_pow * dl));
        if dl < dl_min
            warning('nlng_smoothupdatebyparticle:ErrorTolerance', 'Minimum step size reached. Local error tolerance exceeded.');
            ppsl_prob = 1E10;
            break;
        end
    end
    
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
    

    
end


if display.plot_particle_paths
    figure(10)
    plot(state_evo(1,:), state_evo(3,:), ':');
    plot(init_state(1), init_state(3), 'o');
    plot(state(1), state(3), 'xr', 'markersize', 8);
end


end



