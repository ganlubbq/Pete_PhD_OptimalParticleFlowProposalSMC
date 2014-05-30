clup
dbstop if error
dbstop if warning

rand_seed = 0;

% Set random seed
s = RandStream('mt19937ar', 'seed', rand_seed);
RandStream.setDefaultStream(s);

% Parameters
nu = 3;
K = 100;
lambda = 20;
N = 1000;
nu_lim = 50;

% Create some data by simulating from a student-t distribution
observ = trnd(nu, [1,K]);

% Sample the prior
nu_rng = 0:nu_lim;
nu_prior = poissrnd(lambda, [N,1]);
nu_prior_prob = log(poisspdf(nu_prior, lambda));
nu_arr = nu_prior;
weight = zeros(N,1);
inc_w = nu_prior_prob;

% Artificial time discretisation
lam_rng = 0:0.01:1;
L = length(lam_rng);

% Probabilities
for kk = 1:length(nu_rng), prior_rng(kk) = log(poisspdf(nu_rng(kk),lambda)); end
for kk = 1:length(nu_rng), lhood_rng(kk) = sum(log(tpdf(observ,nu_rng(kk)))); end
lhood_rng(1) = -inf;

% Set random seed
rand_seed = 0;
s = RandStream('mt19937ar', 'seed', rand_seed);
RandStream.setDefaultStream(s);

% Loop through artificial time
for ll = 1:L-1
    
    % Time
    lam = lam_rng(ll);
    dl = lam_rng(ll+1) - lam_rng(ll);
    
    fprintf(1,'Artificial time step %u, time %f.\n', ll, lam);
    
%     figure(1), hist(nu_arr,0:nu_lim);
    
    % Sample particles
    anc = sample_weights(weight, N, 2);
%     anc = 1:N;
    
    post_rng = prior_rng+lam*lhood_rng;
    post_rng(isnan(post_rng)) = -inf;
    post_rng = post_rng - logsumexp(post_rng,2);
%     pt_prob_rng = exp(post_rng);
    
    % Histogram states
    count = hist(nu_arr, nu_rng);
    pt_prob_rng = count/N;
    
    % Arrays
    last_nu_arr = nu_arr; nu_arr = zeros(size(nu_arr));
    last_inc_w = inc_w; inc_w = zeros(size(inc_w));
    last_weight = weight;
    jump_count = 0;
    
    % Log-likelihoods
    max_lhood = max(lhood_rng(last_nu_arr));
    
    % Particle loop
    for ii = 1:N
        
        % Get things
        nu = last_nu_arr(anc(ii));
        last_nu = nu;
        lhood = lhood_rng(nu==nu_rng);
        
        % Exit rate
        pt_prob = pt_prob_rng(nu==nu_rng);
        exit_rate = -(1-pt_prob).*(lhood-max_lhood-0.1);
%         exit_rate = 10000;
        
        % See if there should be a transition in the next dl
        hold_time = exprnd(1/exit_rate);
        
        if hold_time < dl
            
            latest_lam = lam;
             
                %%% MAJOR ASSUMPTION HERE - THAT ONLY ONE JUMP OCCURS
                %%% IN THE dl INTERVAL.
                
                p_change = post_rng;
                p_change(nu==nu_rng) = -inf;
                p_change(nu_rng<(nu-1)) = -inf;
                p_change(nu_rng>(nu+1)) = -inf;
                nu_idx = sample_weights(p_change,1,2);
                nu = nu_rng(nu_idx);
                
%                 p_change = ones(1,2)/2;

% %                 Calculate transition probabilities
%                 p_change(1) = log(poisspdf(nu+1,lambda)) + lam*sum(log(tpdf(observ,nu+1)));
%                 p_change(2) = log(poisspdf(nu-1,lambda)) + lam*sum(log(tpdf(observ,nu-1)));
%                 if nu == 1
%                     p_change(2) = -inf;
%                 end
%                 p_change = p_change - logsumexp(p_change,2);
%                 
%                 % Sample a new state
%                 if log(rand) < p_change(1)
%                     nu = nu + 1;
%                 else
%                     nu = nu - 1;
%                 end
                
                jump_count = jump_count + 1;
                
%                 latest_lam = lam + hold_time;
%                 hold_time = exprnd(1/exit_rate);
%                 assert( (latest_lam+hold_time)>(lam+dl) );
            

        
        end
        
        % Store particle
        nu_arr(ii) = nu;
        
        % Weight particle
%         inc_w(ii) = prior_rng(nu==nu_rng) + lam*lhood_rng(nu==nu_rng);
%         weight(ii) = inc_w(ii) - last_inc_w(anc(ii));
%         weight(ii) = last_weight(anc(ii)) + inc_w(ii) - last_inc_w(anc(ii));
%         weight(ii) = 0;
        
        lin_oid_rng = exp(prior_rng + lam*lhood_rng);
        if nu ~= last_nu
            if nu > 0
                w_up1 =   lin_oid_rng( (last_nu-1)==nu_rng ) / ( lin_oid_rng( (nu-2)==nu_rng ) + lin_oid_rng( (nu)==nu_rng ) );
            else
                w_up1 =   0;
            end
            w_down1 = lin_oid_rng( (last_nu+1)==nu_rng ) / ( lin_oid_rng( (nu)==nu_rng ) + lin_oid_rng( (nu+2)==nu_rng ) );
            weight(ii) = -log( w_up1 + w_down1 );
        else
            weight(ii) = 0;
        end
        weight(ii) = weight(ii) + prior_rng(nu==nu_rng) + lam*lhood_rng(nu==nu_rng);
        
    end
    
    calc_ESS(weight)
%     mean(nu_arr)
    jump_count
    
end

% Resample
[anc] = sample_weights(weight,N,2);
nu_arr = nu_arr(anc);

% Estimate posterior probability.
count = hist(nu_arr, nu_rng);
pt_prob_rng = count/N;

figure(1), hist(nu_arr, 0:20);


%%
post_prob_rng = zeros(size(nu_rng));
for kk = 1:length(nu_rng), prior_rng(kk) = log(poisspdf(nu_rng(kk),lambda)); end
for kk = 1:length(nu_rng), lhood_rng(kk) = sum(log(tpdf(observ,nu_rng(kk)))); end
post_rng = prior_rng+lhood_rng;
post_rng(1) = -inf;
post_rng = post_rng - logsumexp(post_rng,2);
% figure, hold on,
% plot(nu_rng, prior_rng, 'b')
% plot(nu_rng, lhood_rng, 'r')
figure, hold on
plot(nu_rng, exp(prior_rng), 'b');
plot(nu_rng, exp(post_rng), 'm');
plot(nu_rng, pt_prob_rng, ':g');
figure, plot(histc(nu_arr,0:nu_lim)/N-exp(post_rng'))

%%
% Ntest = N;
% test = nu_rng(sample_weights(post_rng,Ntest,1));
% figure, plot(histc(test,0:nu_lim)/Ntest-exp(post_rng))
