function [ state, weight ] = lg_smoothupdate( display, algo, model, fh, obs, prev_state, weight)
%lg_smoothupdate Apply a smooth update for the linear Gaussian model.

% Variables
y = obs;
H = model.H;
R = model.R;

% Arrays
state = zeros(model.ds, algo.N);

% Particle loop
for ii = 1:algo.N
    
    % What prior?
    if isempty(prev_state)
        m = model.m1;
        P = model.P1;
    else
        m = model.A*prev_state(:,ii);
        P = model.Q;
    end
    
    % Sample prior
        if ~isempty(prev_state)
            [x0, ppsl_prob] = feval(fh.transition, model, prev_state(:,ii));
        else
            [x0, ppsl_prob] = feval(fh.stateprior, model);
        end
    
    % Analytical flow
    [ x, wt_jac, prob_ratio] = linear_flow_move( 1, 0, x0, m, P, y, H, R, algo.Dscale );
    
    % Store state
    state(:,ii) = x;
    
    % Densities
    if ~isempty(prev_state)
        [~, trans_prob] = feval(fh.transition, model, prev_state(:,ii), state(:,ii));
    else
        [~, trans_prob] = feval(fh.stateprior, model, state(:,ii));
    end
    [~, lhood_prob] = feval(fh.observation, model, state(:,ii), obs);

    % Weight update
    if algo.Dscale == 0
        weight(ii) = weight(ii) + lhood_prob + trans_prob - ppsl_prob + log(wt_jac);
    else
        weight(ii) = weight(ii) + lhood_prob + trans_prob - ppsl_prob + prob_ratio;
    end

end

end
