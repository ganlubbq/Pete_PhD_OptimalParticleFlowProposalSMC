function [ state, prob ] = sinusoidseparation_PFstateproposal( algo, model, prev_kk, prev_state, observ )
% Sample and/or calculate proposal density for the sinusoidal separation 
%model. This uses the particle flow approximation to the OID.

% figure(1), hold on

% Split state into continuous and discrete components
prev_con_state = prev_state(1:model.dsc,1);
prev_dis_state = prev_state(model.dsc+1:model.ds,1);

% State prior
prior_mn = sinusoidseparation_f(model, prev_kk, prev_con_state);

% Sample it
[state, prob] = sinusoidseparation_transition(model, prev_kk, prev_state);

% Split state into continuous and discrete components
con_state = state(1:model.dsc,1);
dis_state = state(model.dsc+1:model.ds,1);

% Solve ODE
lam_rng = [0 1];
% options.Jacobian = @(lam,x)flow_jacobian(model, lam, x);
% options.Jacobian = [];
if dis_state(end) == 0
    options = odeset('Jacobian', @(lam,x)flow_jacobian(model, lam, x, dis_state), 'RelTol', 1E-2);
else
    options = odeset('RelTol', 1E-2);
end
[lam, x] = ode23(@(lam_in, x_in) calc_particle_velocity(model, lam_in, x_in, observ, prior_mn, dis_state), lam_rng, con_state, options);
con_state = x(end,:)';
state = [con_state; dis_state];

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

function v = calc_particle_velocity(model, lam, x, y, m, dis_state)

% Shorter variable names
Q = model.Q;
R = model.R;
ds = model.dsc;
do = model.do;

% Linearise
H = sinusoidseparation_obsjacobian(model, x, dis_state);

if dis_state(end) == 0

    % Find particle velocity using Gaussian approximation
    y_mod = y - sinusoidseparation_h(model, x, dis_state) + H*x;
    A = -0.5*Q*H'*((R+lam*H*Q*H')\H);
    b = (eye(ds)+2*lam*A)*((eye(ds)+lam*A)*Q*H'*(R\y_mod)+A*m);
    v = A*x+b;

elseif dis_state(end) == 1
    
    % Find particle velocity using incompressible flow FOR MV CAUCHY NOISE
    y_mn = sinusoidseparation_h(model, x, dis_state);
    dlogh_dx = (1+do)*H'*(R\(y-y_mn))/(1+(y-y_mn)'*(R\(y-y_mn)));
    dlogp_dx = lam*dlogh_dx - Q\(x-m);
    loglhood = log(mvtpdf((y-y_mn)', R, 1));
    norm_const = 0.01*ds + (dlogp_dx'*dlogp_dx);
    corr_mat = eye(ds) - dlogp_dx*dlogp_dx'/norm_const;
    v = -loglhood*dlogp_dx/norm_const;% - corr_mat*x;

end

end

function J = flow_jacobian(model, lam, x, dis_state)

% Shorter variable names
Q = model.Q;
R = model.R;

% Linearise
H = sinusoidseparation_obsjacobian(model, x, dis_state);

% Jacobian
J = -0.5*Q*H'*((R+lam*H*Q*H')\H);

end

