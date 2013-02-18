clup
dbstop if error
dbstop if warning


% Make an artificial signal
load('template_beat.mat');
signal = [template(21:end) zeros(1,5) template zeros(1,3) template(1:12)];
signal = signal + mvnrnd(0, 0.01, length(signal))';

model.K = length(signal);
model.dw = length(template);

tmp = template; clear template;
template.m = tmp'; clear tmp;
template.P = 0.5*eye(model.dw);

% Some parameters
model.A_mn = 1;
model.A_vr = 0.01;
model.tau_a = 2;
model.tau_b = 0.6;
model.tau_s = 0.2;
model.fs = 30;
model.y_obs_vr = 0.2^2;

% Some settings
algo.N = 100;

% Arrays
prior_tau = zeros(algo.N,1);
posterior_tau = zeros(algo.N,1);
prior_A = zeros(algo.N,1);
posterior_A = zeros(algo.N,1);
bs_weight = zeros(algo.N,1);
pfp_weight = zeros(algo.N,1);

% Time
t = (0:model.K-1)'/model.fs;

% % Plot
% figure, plot(t, signal)

tic

% Particle loop
for ii = 1:algo.N
    
    % Sample prior
    tau = model.tau_s + invgamrnd(model.tau_a, model.tau_b);
    A = mvnrnd(model.A_mn, model.A_vr);
    ppsl_prob = log(invgampdf(tau-model.tau_s, model.tau_a, model.tau_b)) ...
               +loggausspdf(A, model.A_mn, model.A_vr);
    lhood = likelihood( model, tau, A, template, signal );
    bs_weight(ii) = lhood;
    
    % Store
    prior_tau(ii) = tau;
    prior_A(ii) = A;
    
    % Particle flow
%     lam_rng = 0:0.01:1;
    lam_rng = [0 1];
%     options = odeset('Jacobian', @(lam, x_in) flow_jacobian(model, lam, x, template), 'RelTol', 1E-1);
    options = odeset('RelTol', 1E-3);
    [lam, x] = ode45(@(lam_in, x_in) flow_velocity(model, lam_in, x_in, template, signal), lam_rng, [tau; A], options);
    tau = x(end,1);
    A = x(end,2);
    
    figure(1), hold on, plot(lam, x)
%     figure(2), hold on, plot(x(:,1), x(:,2)), plot(x(end,1), x(end,2),'o')
    
    % Weight
    lhood = likelihood( model, tau, A, template, signal );
    if (tau>model.tau_s)&&(A>0)
        trans_prob = log(invgampdf(tau-model.tau_s, model.tau_a, model.tau_b)) ...
            + loggausspdf(A, model.A_mn, model.A_vr);
    else
        trans_prob = -inf;
    end
    pfp_weight(ii) = lhood + trans_prob - ppsl_prob;
    
    % Store
    posterior_tau(ii) = tau;
    posterior_A(ii) = A;
    
end

toc

%% Analysis

% ESS
bs_ESS = calc_ESS(bs_weight)
pfp_ESS = calc_ESS(pfp_weight)

% Means
bs_tau_mn  = exp(normalise_weights(bs_weight)) *prior_tau
pfp_tau_mn = exp(normalise_weights(pfp_weight))*posterior_tau
bs_A_mn  = exp(normalise_weights(bs_weight)) *prior_A
pfp_A_mn = exp(normalise_weights(pfp_weight))*posterior_A
