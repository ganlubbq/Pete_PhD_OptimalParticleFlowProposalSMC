function [ state, prob ] = linearGaussian_PFstateproposal( algo, model, prev_kk, prev_state, observ )
%NLBENCHMARK_STATEPROPOSAL Sample and/or calculate proposal density for
%2D tracking. This uses the particle flow approximation to the OID.

% State prior
prior_mn = model.A*prev_state;

% Sample it
[state, prob] = linearGaussian_transition(model, prev_kk, prev_state);

% if prev_kk > 2
%     th = 0:pi/720:2*pi;
%     [X, Y] = pol2cart(th, (observ(2)+2*sqrt(model.R(2,2)))*ones(size(th)));
%     figure(1), hold on
%     plot(X, Y, '--r')
% end

% x = state;
% lam = 0;
% 
% Flow integration loop
% for ll = 1:algo.L
% while lam < 1
%     
%     % Find velocity
%     v = calc_particle_velocity(model, lam, x, observ, prior_mn);
%     
%     % Choose step size
%     vy = model.H*v;
%     ry = observ - model.H*x;
%     dl = 0.1*magn(ry./vy);
%     dl = min(dl, 1/algo.L);
%     dl = min(dl, 1-lam);
%     
% %     % Euler update
% %     x = x+v*dl;
%     
%     % Runge-Kutta update
%     v = calc_particle_velocity(model, lam, x, observ, prior_mn);
%     dx1 = v*dl;
%     x1 = x + 0.5*dx1;
%     v1 = calc_particle_velocity(model, lam+dl/2, x1, observ, prior_mn);
%     dx2 = v1*dl;
%     x2 = x + 0.5*dx2;
%     v2 = calc_particle_velocity(model, lam+dl/2, x2, observ, prior_mn);
%     dx3 = v2*dl;
%     x3 = x + dx3;
%     v3 = calc_particle_velocity(model, lam+dl, x3, observ, prior_mn);
%     dx4 = v3*dl;
%     x = x + (dx1 + 2*dx2 + 2*dx3 + dx4)/6;
%     
%     % Update pseudo-time
%     lam = lam + dl;
%     
%     
% %     % Step size
% %     dl = 1/algo.L;
% %     lam = (ll-1)*dl;
% %     lam = (1-2^-(ll-1));
% %     dl = (1-2^-(ll)) - (1-2^-(ll-1));
% %     if ll == algo.L
% %         dl = 2^-(ll-1);
% %     end
%     
% %     % Predict
% %     v = calc_particle_velocity(model, lam, state, observ, prior_mn);
% %     pred_state = state + v*dl;
% %     
% % %     % Update
% %     for ii = 1:1
% %         pred_v = calc_particle_velocity(model, lam+dl, pred_state, observ, prior_mn);
% %         pred_state = state + 0.5*dl*(v+pred_v);
% %     end
% 
% %     state = pred_state;
% 
% %     % Runge-Kutta update
% %     v = calc_particle_velocity(model, lam, state, observ, prior_mn);
% %     dx1 = v*dl;
% %     x1 = state + 0.5*dx1;
% %     v1 = calc_particle_velocity(model, lam+dl/2, x1, observ, prior_mn);
% %     dx2 = v1*dl;
% %     x2 = state + 0.5*dx2;
% %     v2 = calc_particle_velocity(model, lam+dl/2, x2, observ, prior_mn);
% %     dx3 = v2*dl;
% %     x3 = state + dx3;
% %     v3 = calc_particle_velocity(model, lam+dl, x3, observ, prior_mn);
% %     dx4 = v3*dl;
%     
% %     state = state + (dx1 + 2*dx2 + 2*dx3 + dx4)/6;
%     
% %     if prev_kk > 2
% %         figure(1), hold on, plot(state(1), state(2), 'xb')
% %     end
%     
% end
% 
% state = x;

lam = [0 1];
options.Jacobian = @(lam,x)flow_jacobian(model, lam);
[lam, x] = ode45(@(lam_in, x_in) calc_particle_velocity(model, lam_in, x_in, observ, prior_mn), lam, state, options);

state = x(end,:)';

end

function v = calc_particle_velocity(model, lam, x, y, m)

% Shorter variable names
Q = model.Q;
R = model.R;
ds = model.ds;
H = model.H;

% Find particle velocity using Gaussian approximation
A = -0.5*Q*H'*((R+lam*H*Q*H')\H);
b = (eye(ds)+2*lam*A)*((eye(ds)+lam*A)*Q*H'*(R\y)+A*m);
v = A*x+b;

% new_x = - (A'*A)\(A'*b);
% v = new_x-x;

end

function J = flow_jacobian(model, lam)

% Shorter variable names
Q = model.Q;
R = model.R;
H = model.H;

% Jacobian
J = -0.5*Q*H'*((R+lam*H*Q*H')\H);

end
