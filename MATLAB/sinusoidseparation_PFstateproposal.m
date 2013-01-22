function [ state, prob ] = nlbenchmark_PFstateproposal( algo, model, prev_kk, prev_state, observ )
%NLBENCHMARK_STATEPROPOSAL Sample and/or calculate proposal density for
%nonlinear benchmark. This uses the particle flow approximation to the OID.

% figure(1), hold on

% State prior
prior_mn = nlbenchmark_f(model, prev_kk, prev_state);

% Sample it
[state, prob] = nlbenchmark_transition(model, prev_kk, prev_state);

% Solve ODE
lam_rng = [0 1];
% options.Jacobian = @(lam,x)flow_jacobian(model, lam, x);
options.Jacobian = [];
[lam, x] = ode45(@(lam_in, x_in) calc_particle_velocity(model, lam_in, x_in, observ, prior_mn), lam_rng, state, options);
state = x(end,:)';

% lam = 0;
% dl = 1/algo.L;
% x = state;
% y = observ;
% m = prior_mn;
% for ll = 1:algo.L
%     
% %     ll
%     
%     v = calc_particle_velocity(model, lam, x, y, m);
%     x = x + v*dl;
%     
%     lam = lam + dl;
%     
% %     if prev_kk>4
% %         figure(1); plot(lam, x(1), 'xr');
% % %         pause
% %     end
%     
%     last_x = x;
%     last_v = v;
%     
% end
% state = x;

end

function v = calc_particle_velocity(model, lam, x, y, m)

% Shorter variable names
Q = model.Q;
R = model.R;
ds = model.ds;
do = model.do;

% Linearise
H = nlbenchmark_obsjacobian(model, x);

% Find particle velocity using Gaussian approximation
y_mod = y - nlbenchmark_h(model, x) + H*x;
A = -0.5*Q*H'*((R+lam*H*Q*H')\H);
b = (eye(ds)+2*lam*A)*((eye(ds)+lam*A)*Q*H'*(R\y_mod)+A*m);
v = A*x+b;
    
%     % Find particle velocity using Incompresible flow
%     y_mn = nlbenchmark_h(model, x);
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
%     y_mn = nlbenchmark_h(model, x);
%     dlogp_dx = lam*H'*(R\(y-y_mn)) - Q\(x-m);
%     ddlogp_ddx = -(lam*H'*(R\H) + inv(Q));
%     dloglhood_dx = H'*(R\(y-y_mn));
%     v = -ddlogp_ddx \ ( dloglhood_dx + A'*dlogp_dx );

%     % Find particle velocity using incompressible flow FOR MV CAUCHY NOISE
%     y_mn = nlbenchmark_h(model, x);
%     dlogh_dx = (1+do)*H'*(R\(y-y_mn))/(1+(y-y_mn)'*(R\(y-y_mn)));
%     dlogp_dx = lam*dlogh_dx - Q\(x-m);
%     loglhood = log(mvtpdf((y-y_mn)', R, 1));
%     norm_const = 0.01*ds + (dlogp_dx'*dlogp_dx);
% %     corr_mat = eye(ds) - dlogp_dx*dlogp_dx'/norm_const;
%     v = -loglhood*dlogp_dx/norm_const;% - corr_mat*x;



end

function J = flow_jacobian(model, lam, x)

% Shorter variable names
Q = model.Q;
R = model.R;

% Linearise
H = nlbenchmark_obsjacobian(model, x);

% Jacobian
J = -0.5*Q*H'*((R+lam*H*Q*H')\H);

end

