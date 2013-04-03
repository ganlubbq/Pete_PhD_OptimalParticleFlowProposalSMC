function [ state, weight, state_evolution ] = lg_smoothupdate( display, algo, model, fh, obs, prev_state, weight)
%lg_smoothupdate Apply a smooth update for the linear Gaussian model.

% Prior density
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
num_steps = 200;
scale_fact = (1-ratio)/(ratio*(1-ratio^num_steps));
lam_rng = cumsum([0 scale_fact*ratio.^(1:num_steps)]);
L = length(lam_rng);

% State evolution array (in case we want to plot the trajectories)
state_evolution = zeros(model.ds, algo.N, L);
state_evolution(:,:,1) = state;

% Prior
if isempty(prev_state)
    prior_mn = repmat(model.m1, 1, algo.N);
    P = model.P1;
else
    prior_mn = model.A*prev_state;
    P = model.Q;
end

% Other variables
y = obs;
H = model.H;
R = model.R;
I = eye(model.ds);

% Jacobian integral for weight
wt_jac = ones(1, algo.N);

% Pseudo-time loop
for ll = 1:L-1
    
    % Pseudo-time
    lam = lam_rng(ll);
    dl = lam_rng(ll+1)-lam_rng(ll);
    
    % Particle loop
    for ii = 1:algo.N
        
        % Get state
        x = state(:,ii);
        m = prior_mn(:,ii);
        
        % Calculate velocity
        [ A, b ] = linear_flow( lam, m, P, y, H, R );
        v = A*x + b;
        
        % Push forward
        x = x + v*dl;
        
        % Stochastic bit
        if algo.flag_stochastic
            x = mvnrnd(x', 2*dl*algo.D)';
        end
        
        % Store state
        state(:,ii) = x;
        state_evolution(:,ii,ll+1) = x;
        
        % Update weight
        wt_jac(ii) = wt_jac(ii) * det(I + dl*A);
        
    end
    
end

% Weight update loop
for ii = 1:algo.N
    
    % Densities
    if ~isempty(prev_state)
        [~, trans_prob] = feval(fh.transition, model, prev_state(:,ii), state(:,ii));
    else
        [~, trans_prob] = feval(fh.stateprior, model, state(:,ii));
    end
    [~, lhood_prob] = feval(fh.observation, model, state(:,ii), obs);

    % Weight update
    weight(ii) = weight(ii) + lhood_prob + trans_prob - init_trans_prob(ii) + log(wt_jac(ii));

end

% Plot particle paths (first state only)
if display.plot_particle_paths
    figure(1), clf, hold on
    for ii = 1:algo.N
        plot(lam_rng, squeeze(state_evolution(1,ii,:)));
    end
end

end
