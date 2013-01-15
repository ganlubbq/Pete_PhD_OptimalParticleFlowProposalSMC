function [ state, prob ] = nlbenchmark_PFstateproposal( algo, model, prev_kk, prev_state, observ )
%NLBENCHMARK_STATEPROPOSAL Sample and/or calculate proposal density for
%nonlinear benchmark. This uses the particle flow approximation to the OID.

figure(1), hold on

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
g = sqrt(model.sigy);
x = state;

% Flow integration loop
for ll = 1:algo.L
    
    % How far through?
    lam = (ll-1)*dl;
    
    % Linearise observation about this point
    H = model.alpha1 * model.alpha2 * sign(x) * abs(x)^(model.alpha2 - 1);
    H(isnan(H)) = 0;
    
    % Find particle velocity using Gaussian approximation
    y_mod = y - nlbenchmark_h(model, x) + H*x;
    A = -0.5*Q*H'*((R+lam*H*Q*H')\H);
    b = (1+2*lam*A)*((1+lam*A)*Q*H'*(R\y_mod)+A*m);
    v = A*x+b;

%     % Find particle velocity using Incompresible flow
%     y_mn = nlbenchmark_h(model, x);
%     dlogp_dx = lam*H'*(R\(y-y_mn)) - Q\(x-m);
%     loglhood = loggausspdf(y, y_mn, R);
%     norm_const = (dlogp_dx'*dlogp_dx);
%     corr_mat = eye(ds) - dlogp_dx*dlogp_dx'/norm_const;
%     v = -loglhood*dlogp_dx/norm_const - corr_mat*x;

%     % Find particle velocity using Incompresible flow FOR CAUCHY LIKELIHOOD
%     y_mn = nlbenchmark_h(model, x);
%     dlogp_dx = - Q\(x-m) - lam*( 2*g*(y-y_mn) - g*H )/( R + (y-y_mn)^2 );
%     loglhood = log( tpdf( (y-y_mn)/g, 1) );
%     norm_const = (dlogp_dx'*dlogp_dx);
%     corr_mat = 1 - dlogp_dx*dlogp_dx'/norm_const;
%     if norm_const > 1E-6
%         v = -loglhood*dlogp_dx/norm_const - corr_mat*x;
%     else
%         v = 0;
%     end
    
    % Find new state
    x = x + v*dl;
    
    plot(ll, x);
    
end

end

