function [ state, weight ] = lg_smoothupdate( display, algo, model, fh, obs, prev_state, weight)
%lg_smoothupdate Apply a smooth update for the linear Gaussian model.

% Variables
y = obs;
H = model.H;
R = model.R;

% What prior?
if isempty(prev_state)
    P = model.P1;
else
    P = model.Q;
end

% Integrated flow matrixes
[ x0_mat, m_mat, Hy_vec, x_cov, wt_jac ] = lg_linearflowmatrixes( P, H, R, y, algo.Dscale );

% Arrays
state = zeros(model.ds, algo.N);

% Particle loop
for ii = 1:algo.N
    
    % What prior?
    if isempty(prev_state)
        m = model.m1;
    else
        m = model.A*prev_state(:,ii);
    end
    
    % Sample prior
    if ~isempty(prev_state)
        [x0, prior_prob] = feval(fh.transition, model, prev_state(:,ii));
    else
        [x0, prior_prob] = feval(fh.stateprior, model);
    end
    
    % Analytical flow using pre-calculated matrixes
    [ x, ppsl_prob] = lg_move( x0, m, P, x0_mat, m_mat, Hy_vec, x_cov );
    
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
        weight(ii) = weight(ii) + lhood_prob + trans_prob - prior_prob + log(wt_jac);
    else
        weight(ii) = weight(ii) + lhood_prob + trans_prob - ppsl_prob;
    end

end

end
