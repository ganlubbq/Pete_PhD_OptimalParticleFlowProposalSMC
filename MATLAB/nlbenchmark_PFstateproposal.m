function [ state, prob ] = nlbenchmark_PFstateproposal( algo, model, prev_kk, prev_state, observ )
%NLBENCHMARK_STATEPROPOSAL Sample and/or calculate proposal density for
%nonlinear benchmark. This uses the particle flow approximation to the OID.

% figure(1), hold on

% State prior
prior_mn = nlbenchmark_f(model, prev_kk, prev_state);

% Sample it
[state, prob] = nlbenchmark_transition(model, prev_kk, prev_state);

% % Step size
% dl = 1/algo.L;
% 
% % Shorter variable names
% Q = model.sigx;
% R = model.sigy;
% m = prior_mn;
% y = observ;
% g = sqrt(model.sigy);
% x = state;

% % Flow integration loop
% for ll = 1:algo.L
%     
%     % How far through?
%     lam = (ll-1)*dl;
%     
%     % Linearise observation about this point
%     H = model.alpha1 * model.alpha2 * sign(x) * abs(x)^(model.alpha2 - 1);
%     H(isnan(H)) = 0;
%     
%     % Find particle velocity using Gaussian approximation
%     y_mod = y - nlbenchmark_h(model, x) + H*x;
%     A = -0.5*Q*H'*((R+lam*H*Q*H')\H);
%     b = (1+2*lam*A)*((1+lam*A)*Q*H'*(R\y_mod)+A*m);
%     v = A*x+b;
% 
% %     % Find particle velocity using Incompresible flow
% %     y_mn = nlbenchmark_h(model, x);
% %     dlogp_dx = lam*H'*(R\(y-y_mn)) - Q\(x-m);
% %     loglhood = loggausspdf(y, y_mn, R);
% %     norm_const = (dlogp_dx'*dlogp_dx);
% %     corr_mat = eye(ds) - dlogp_dx*dlogp_dx'/norm_const;
% %     v = -loglhood*dlogp_dx/norm_const - corr_mat*x;
% 
% %     % Find particle velocity using Incompresible flow FOR CAUCHY LIKELIHOOD
% %     y_mn = nlbenchmark_h(model, x);
% %     dlogp_dx = - Q\(x-m) - lam*( 2*g*(y-y_mn) - g*H )/( R + (y-y_mn)^2 );
% %     loglhood = log( tpdf( (y-y_mn)/g, 1) );
% %     norm_const = (dlogp_dx'*dlogp_dx);
% %     corr_mat = 1 - dlogp_dx*dlogp_dx'/norm_const;
% %     if norm_const > 1E-6
% %         v = -loglhood*dlogp_dx/norm_const - corr_mat*x;
% %     else
% %         v = 0;
% %     end
%     
%     % Find new state
%     x = x + v*dl;
%     
%     plot(ll, x);
%     
% end

lam_rng = [0 1];
options.Jacobian = @(lam,x)flow_jacobian(model, lam, x);
[lam, x] = ode45(@(lam_in, x_in) calc_particle_velocity(model, lam_in, x_in, observ, prior_mn), lam_rng, state, options);

state = x(end,:)';

end

function v = calc_particle_velocity(model, lam, x, y, m)

% Shorter variable names
Q = model.sigx;
R = model.sigy;
ds = model.ds;

% Linearise
H = nlbenchmark_obsjacobian(model, x);

% Find particle velocity using Gaussian approximation
y_mod = y - nlbenchmark_h(model, x) + H*x;
A = -0.5*Q*H'*((R+lam*H*Q*H')\H);
b = (eye(ds)+2*lam*A)*((eye(ds)+lam*A)*Q*H'*(R\y_mod)+A*m);
v = A*x+b;
    
%     % Find particle velocity using Incompresible flow
%     y_mn = tracking_h(model, x);
%     dlogp_dx = lam*H'*(R\(y-y_mn)) - Q\(x-m);
%     loglhood = loggausspdf(y, y_mn, R);
%     norm_const = (dlogp_dx'*dlogp_dx);
%     corr_mat = eye(ds) - dlogp_dx*dlogp_dx'/norm_const;
% %     if norm_const > 0.001;
%         v = -loglhood*dlogp_dx/norm_const - corr_mat*x;
% %     else
% %         v = zeros(model.ds, 1);
% %     end

%     % Find particle velocity using small curvature flow
%     A = -0.5*Q*H'*((R+lam*H*Q*H')\H);
%     y_mn = tracking_h(model, x);
%     dlogp_dx = lam*H'*(R\(y-y_mn)) - Q\(x-m);
%     ddlogp_ddx = -(lam*H'*(R\H) + inv(Q));
%     dloglhood_dx = H'*(R\(y-y_mn));
%     v = -ddlogp_ddx \ ( dloglhood_dx + A'*dlogp_dx );


end

function J = flow_jacobian(model, lam, x)

% Shorter variable names
Q = model.Q;
R = model.R;

% Linearise
H = tracking_obsjacobian(x);

% Jacobian
J = -0.5*Q*H'*((R+lam*H*Q*H')\H);

end

