function [ state, prob ] = sinusoidseparation_PFstateproposal( algo, model, prev_kk, prev_state, observ )
% Sample and/or calculate proposal density for the sinusoidal separation 
%model. This uses the particle flow approximation to the OID.

% figure(1), hold on
% figure(2), hold on
% figure(3), hold on, plot(observ,'r')
% figure(4), hold on

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
lam_rng = 0:0.01:1;%[0 1];
if dis_state(end) == 0
    options = odeset('Jacobian', @(lam,x)flow_jacobian(model, lam, x, dis_state), 'RelTol', 1E-2);
    [lam, x] = ode23(@(lam_in, x_in) calc_particle_velocity(model, lam_in, x_in, observ, prior_mn, dis_state), lam_rng, con_state, options);
    con_state = x(end,:)';
    
else
    
%     [ exp_samps, GL_weights ] = gengausslegquadrule( 5 );
%     options = [];
%     
%     starting_con_state = con_state;
%     con_state = zeros(model.dsc,1);
%     
% %     figure(1), clf, hold on
%     
%     for ii = 1:length(exp_samps)
%         
%         xi_ii = 2*exp_samps(ii);
%         aug_state = [starting_con_state; xi_ii];
%         [lam, x] = ode15s(@(lam_in, x_in) calc_SMoN_particle_velocity(model, lam_in, x_in, observ, prior_mn, dis_state), lam_rng, aug_state, options);
%         
%         final_x = x(end,1:end-1)';
%         
%         con_state = con_state + GL_weights(ii) * final_x;
%         
% %         plot(final_x(1), final_x(2), 'bo');
%         
%     end
    
%     ppsl_rate = 2;
%     xi = exprnd(ppsl_rate);
%     prob = prob + log(gampdf(xi,2,2))-log(exppdf(xi,ppsl_rate));
    
    % Sample a mixing variable
    xi = chi2rnd(model.dfy);
    aug_state = [con_state; xi];
    
    options = [];
    [lam, x] = ode15s(@(lam_in, x_in) calc_SMoN_particle_velocity(model, lam_in, x_in, observ, prior_mn, dis_state), lam_rng, aug_state, options);
    con_state = x(end,1:end-1)';
    
%     options = [];
%     [lam, x] = ode15s(@(lam_in, x_in) calc_particle_velocity(model, lam_in, x_in, observ, prior_mn, dis_state), lam_rng, con_state, options);
%     con_state = x(end,:)';
    
%     if prev_kk > 0
%         figure(1), plot(x(:,1), x(:,2)), plot(x(end,1), x(end,2), 'bo')
%         figure(2), plot(x(:,2), x(:,3)), plot(x(end,2), x(end,3), 'bo')
%         figure(3), pred_obs = sinusoidseparation_h(model, con_state, dis_state), plot(pred_obs, 'b');
%         figure(4), plot(lam, log(x(:,end)))
%     end
    
%     % Solve ODE for sample AND UT approximation in parallel, using UT
%     % approximation to expected log-likelihood
%     
%     % UT approximation of (Gaussian) prior
%     [utw, ~, c] = ut_weights(model.dsc, [], [], 1);
%     sig_pts = ut_sigmas(prior_mn, model.Q, c);
%     
%     % Stack up the sigma points
%     con_stack = [con_state; sig_pts(:)];
%     
%     % Give it to the solver
%     options = [];
%     [lam, x] = ode15s(@(lam_in, x_in) calc_sigset_particle_velocity(model, lam_in, x_in, observ, prior_mn, utw, dis_state), lam_rng, con_stack, options);
%     
% %     if prev_kk > 0
% %         plot(x(:,1), x(:,2))
% %         plot(x(end,1), x(end,2), 'bo')
% %     end
%     
%     % Get the right bit
%     con_state = x(end,1:model.dsc)';
    
end

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
    loglhood = log(mvcauchypdf(y', y_mn', R));
    norm_const = 0.01*ds + (dlogp_dx'*dlogp_dx);
%     corr_mat = eye(ds) - dlogp_dx*dlogp_dx'/norm_const;
    v = -loglhood*dlogp_dx/norm_const;% - corr_mat*x;

end

end

function v = calc_SMoN_particle_velocity(model, lam, x, y, m, dis_state)

% Shorter variable names
Q = model.Q;
R = model.R;
ds = model.dsc;
do = model.do;

xi = x(end);
x(end) = [];

% Linearise
H = sinusoidseparation_obsjacobian(model, x, dis_state);
y_mod = y - sinusoidseparation_h(model, x, dis_state) + H*x;

% Useful intermediates
D  = (y-H*x)'*(R\(y-H*x));
% Dp = (y-H*m)'*(R\(y-H*m));
% D  = (y_mod-H*x)'*(R\(y_mod-H*x));

% Some constants
ga = 1+0.5*(lam*do-1);
gb = (1+lam*D)/2;

% Set xi deterministically??
% xi = gamrnd(ga, 1/gb);
% xi = ga/gb;

% Normal state velocity
Rxi = R/xi;
A = -0.5*Q*H'*((Rxi+lam*H*Q*H')\H);
b = (eye(ds)+2*lam*A)*((eye(ds)+lam*A)*Q*H'*(Rxi\y_mod)+A*m);
vx = A*x+b;

%%% XI BIT %%%

% % Some constants
% g1 = -do/2;
% g2 = D/2;
% g3 = -0.5*D*ga/gb + 0.5*do*( psi(ga) - log(gb) );
% 
% % Integral terms
% I2 =  g2 * gb^(-ga-1) * gamma(ga+1)*gammainc(ga+1,gb*xi);
% I3 =  g3 * gb^(-ga) * gamma(ga)*gammainc(ga,gb*xi);
% I1 =  g1 * xi^ga*( log(xi)*(gb*xi)^(-ga)*( gamma(ga)*gammainc(gb*xi,ga) ) - hypergeom([ga ga], [ga+1, ga+1], -gb*xi)/(ga^2) );
% % I1 = 2.2332;
% 
% 
% I = I1 + I2 + I3;
% 
% Nsamp = 1000;
% I1_arr = zeros(1,Nsamp);
% I2_arr = zeros(1,Nsamp);
% I3_arr = zeros(1,Nsamp);
% for ii = 1:Nsamp
%     samp = inf;
%     while samp > xi
%         samp = gamrnd(ga,gb);
%     end
%     I1_arr(ii) = g1*log(samp);
%     I2_arr(ii) = g2*samp;
%     I3_arr(ii) = g3;
% end
% I = (sum(I1_arr)+sum(I2_arr)+sum(I3_arr))*gamma(ga)*gb^(-ga)/Nsamp;
% 
% vxi = xi^(1-ga)*exp(gb*xi)*I;


% if lam > 0
    
    
%     % Expectation
%     samps = exprnd(0.5,[100,1]);
%     I1 = 0; I2 = 0;
%     for ii = 1:length(samps)
%         I1 = I1 + samps(ii)^(-0.5*do*(1-lam))*exp(loggausspdf(y,H*m,H*Q*H'+(R/(lam*samps(ii)))));
%         I1 = I1 + samps(ii)^(-0.5*do*(1-lam))*exp(loggausspdf(y,H*m,H*Q*H'+(R/(lam*samps(ii)))));
%     end
%     sf = det(2*pi*R)^(0.5*(1-lam))*lam^(-do/2);
%     I1 = I1*sf;
%     I2 = I2*sf;
% 
%     
%     % Scale factor
%     g1 = (0.5*y'-lam*b'*H')*(R\y);
%     g2 = -do/2;
%     g3 = 0.5*log(det(2*pi*R)) - trace(A) + b'*(Q\m) + Elogbeta;
%     C = gamma(0.5*lam*do+1)*(0.5*lam*D)^(-0.5*lam*do-1)*( g1*(0.5*lam*do+1)/(0.5*lam*D) + g2 + g3*( log(1/(lam*D)) + psi(0.5*lam*do+1) ) );
%     
%     % Mixing variable velocity
%     vxi = C * xi^(-0.5*lam*do)*exp(0.5*xi*lam*D);
%     
% else
%     
%     % Expectation
%     Elogbeta = -0.5*log(det(2*pi*R)) + 0.5*do*(log(2)-(-psi(1))) - Dp - trace((H'/R)*(H/Q));
%     
%     % Scale factor
%     g1 = 0.5*y'*(R\y);
%     g2 = -do/2;
%     g3 = 0.5*log(det(2*pi*R)) - trace(A) + b'*(Q\m) + Elogbeta;
%     
%     % Mixing variable velocity
%     vxi = 0.5*g1*xi^2 + g2*xi*(log(xi)-1) + g3*xi;
%     
% end

% vxi = xi^(1-ga)*exp(gb*xi);

% % Moment matching
% lDp1 = 1 + lam*D;
% ldp2 = 2 + lam*do;
% mu = ldp2/lDp1;
% A = (do - 4*D - lam*do*D)/( 2*lDp1*ldp2 );
% b = (do - 2*D)/lDp1^2;
% vxi = A*(xi-mu)+b;

% % Approximate?
% xi_mn = ga/gb;
% dtml = xi_mn - xi;
% vxi = sign(dtml)*abs(dtml);

% Fuck it. Just keep it constant
vxi = 0;

v = [vx; vxi];

end

function v = calc_sigset_particle_velocity(model, lam, x, y, m, utw, dis_state)

% Shorter variable names
ds = model.dsc;

% Make sure we're dealing with the Cauchy noise case
assert(dis_state(end) == 1);

% Split out the state and sigma points
state = x(1:ds);
sig_pts = reshape(x(ds+1:end), ds, 2*ds+1);
 
% Find mean log-likelihood using UT approximation
mn_llhood = 0;
for ii = 1:(2*ds+1)
    y_mn = sinusoidseparation_h(model, sig_pts(:,ii), dis_state);
    mn_llhood = mn_llhood + utw(ii) * log(mvcauchypdf(y', y_mn', model.R));
end

v = zeros(size(x));
v(1:ds) = cauchy_velocity(model, lam, state, y, m, mn_llhood, dis_state);
for ii = 1:(2*ds+1)
    v(ii*ds+1:(ii+1)*ds) = cauchy_velocity(model, lam, sig_pts(:,ii), y, m, mn_llhood, dis_state);
end


end


function v = cauchy_velocity(model, lam, x, y, m, mn_llhood, dis_state)

% Shorter variable names
Q = model.Q;
R = model.R;
ds = model.dsc;
do = model.do;

% Linearise
H = sinusoidseparation_obsjacobian(model, x, dis_state);

% Find particle velocity using incompressible flow FOR MV CAUCHY NOISE
y_mn = sinusoidseparation_h(model, x, dis_state);
dlogh_dx = (1+do)*H'*(R\(y-y_mn))/(1+(y-y_mn)'*(R\(y-y_mn)));
dlogp_dx = lam*dlogh_dx - Q\(x-m);
loglhood = log(mvcauchypdf(y', y_mn', R)) - mn_llhood;
norm_const = 0.01*ds + (dlogp_dx'*dlogp_dx);
% corr_mat = eye(ds) - dlogp_dx*dlogp_dx'/norm_const;
v = -loglhood*dlogp_dx/norm_const;% - corr_mat*x;

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

