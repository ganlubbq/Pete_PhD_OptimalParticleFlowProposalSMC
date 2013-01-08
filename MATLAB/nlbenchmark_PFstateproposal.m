function [ state, prob ] = nlbenchmark_PFstateproposal( algo, model, prev_kk, prev_state, observ )
%NLBENCHMARK_STATEPROPOSAL Sample and/or calculate proposal density for
%nonlinear benchmark. This uses the particle flow approximation to the OID.

% State prior
prior_mn = nlbenchmark_f(model, prev_kk, prev_state);

% Sample it
[state, prob] = nlbenchmark_transition(model, prev_kk, prev_state);

% Step size
dl = 1/algo.L;

% Shorter variable names
Q = model.sigx;
R = model.sigy;
m = prior_mn;
y = observ;

% Flow integration loop
for ll = 1:algo.L
    
    % How far through?
    lam = (ll-1)*dl;
    
    % Linearise observation about this point
    H = model.alpha1 * model.alpha2 * sign(state) * abs(state)^(model.alpha2 - 1);
    H(isnan(H)) = 0;
    
    % Find particle velocity using Gaussian approximation
    y_mod = y - nlbenchmark_h(model, state) + H*state;
    A = -0.5*Q*H'*((R+lam*H*Q*H')\H);
    b = (1+2*lam*A)*((1+lam*A)*Q*H'*(R\y_mod)+A*m);
    v = A*state+b;
    
    % Find new state
    state = state + v*dl;
    
end

end

