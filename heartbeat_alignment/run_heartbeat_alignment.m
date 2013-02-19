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
model.tau_s = 0;
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
    
    fprintf('Particle %u.\n',ii)
    
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
    
%     [tau, A, ppsl_prob] = EKF_proposal(model, template, signal);

    D = diag([0.0001 0.0001]);
    x = [tau; A];
    
    lam_rng = [1E-5:1E-5:1E-4 2E-4:1E-4:1E-3 2E-3:1E-3:1E-2 2E-2:1E-2:1];
    x_rng = zeros(2,length(lam_rng));
    w_rng = zeros(1,length(lam_rng));
    for ll = 1:length(lam_rng)-1
        
        lam = lam_rng(ll);
        dl = lam_rng(ll+1)-lam_rng(ll);
        
        v = flow_velocity(model, lam, x, template, signal, D);
        
        x_rng(:,ll) = x;
        
        x = x + v*dl + mvnrnd(zeros(1,2), D*dl)';
        
        w_rng(ll+1) = log(invgampdf(x(1)-model.tau_s, model.tau_a, model.tau_b)) ...
            + loggausspdf(x(2), model.A_mn, model.A_vr) ...
            + lam * likelihood( model, x(1), x(2), template, signal );
        
        if any(x<0)||any(x>100)
            w_rng(ll+2:end) = -inf;
            break;
        end
        
    end
    x_rng(:,end) = x;
    
    figure(2), hold on, plot(x_rng(1,:), x_rng(2,:)), plot(x(1), x(2),'ro')
    figure(3), hold on, plot(lam_rng, w_rng);
    
    w_mat(ii,:) = w_rng;
    
    tau = x(1);
    A = x(2);
    
%     % Particle flow
% %     lam_rng = 0:0.01:1;
%     lam_rng = [0 1];
% %     options = odeset('Jacobian', @(lam, x_in) flow_jacobian(model, lam, x, template), 'RelTol', 1E-1);
%     options = odeset('RelTol', 1E-2);
%     [lam, x] = sdesim(@(lam_in, x_in) flow_velocity(model, lam_in, x_in, template, signal), D, lam_rng, [tau; A], options);
% %     [lam, x] = ode23(@(lam_in, x_in) flow_velocity(model, lam_in, x_in, template, signal), lam_rng, [tau; A], options);
%     tau = x(end,1);
%     A = x(end,2);
    
%     figure(1), hold on, plot(lam, x)
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

tmp = w_mat - max(w_mat(:));
tmp = exp(tmp);
norm_w_mat = w_mat;
for ii = 1:size(w_mat,2), norm_w_mat(:,ii) = tmp(:,ii)/sum(tmp(:,ii)); end
figure, plot(lam_rng, norm_w_mat')
figure, plot(lam_rng, log(norm_w_mat)')

%% Analysis

% ESS
bs_ESS = calc_ESS(bs_weight)
pfp_ESS = calc_ESS(pfp_weight)

% Means
bs_tau_mn  = exp(normalise_weights(bs_weight)) *prior_tau
pfp_tau_mn = exp(normalise_weights(pfp_weight))*posterior_tau
bs_A_mn  = exp(normalise_weights(bs_weight)) *prior_A
pfp_A_mn = exp(normalise_weights(pfp_weight))*posterior_A