% Base script for running particle filter comparisons for SUPF paper

% Structures
% display - parameters for determining MATLAB output
% test - test parameters
% algo - algorithm parameters
% model - model parameters

%% Set Up

if ~exist('test.flag_batch', 'var') || (~test.flag_batch)
    
    clup
    dbstop if error
    % dbstop if warning
    
    %%% SETTINGS %%%
    
    % DEFINE RANDOM SEED
    rand_seed = 0;
    
    % Which model?
    model_flag = 1;     % 1 = linear Gaussian
                        % 2 = contrived nonlinear non-Gaussian
                        % 3 = heartbeat alignment
    
    %%%%%%%%%%%%%%%%
    
    % Set random seed
    rng(rand_seed);
    
    % Set flag to non-batch
    test.flag_batch = false;
    
    % Set function handles
    if model_flag == 1
        addpath('lg');
        fh.setmodel = @lg_setmodel;
        fh.setalgo = @lg_setalgo;
        fh.generatedata = @lg_generatedata;
        fh.transition = @lg_transition;
        fh.observation = @lg_observation;
        fh.stateprior = @lg_stateprior;
        fh.ekfproposal = @lg_oidproposal;
        fh.ukfproposal = @lg_oidproposal;
        fh.linearisedoidproposal = @lg_oidproposal;
        fh.smoothupdate = @lg_smoothupdate;
    elseif model_flag == 2
        addpath('nlng');
        fh.setmodel = @nlng_setmodel;
        fh.setalgo = @nlng_setalgo;
        fh.generatedata = @nlng_generatedata;
        fh.transition = @nlng_transition;
        fh.observation = @nlng_observation;
        fh.stateprior = @nlng_stateprior;
        fh.ekfproposal = @nlng_ekfproposal;
        fh.ukfproposal = @nlng_ukfproposal;
        fh.linearisedoidproposal = @nlng_linearisedoidproposal;
        fh.smoothupdate = @nlng_smoothupdate;
    elseif model_flag == 3
        addpath('ha');
        fh.setmodel = @ha_setmodel;
        fh.setalgo = @ha_setalgo;
        fh.generatedata = @ha_generatedata;
        fh.transition = @ha_transition;
        fh.observation = @ha_observation;
        fh.stateprior = @ha_stateprior;
        fh.ekfproposal = @ha_ekfproposal;
        fh.ukfproposal = @ha_ukfproposal;
        fh.linearisedoidproposal = @ha_linearisedoidproposal;
        fh.smoothupdate = @ha_smoothupdate;
    end
        
    % Set display options
    display.text = true;
    display.plot_during = false;
    if display.plot_during
        display.h_pf(1) = figure;
        display.h_pf(2) = figure;
    end
    display.plot_after = true;
    
    % Set test options
    test.algs_to_run = [1,5];         % Vector of algorithm indexes to run
                                    % 1 = bootstrap
                                    % 2 = EKF proposal
                                    % 3 = UKF proposal
                                    % 4 = linearised OID proposal
                                    % 5 = SUPF
	test.num_filt_pts = 100*ones(5,1);      % Number of particles to use with each algorithm
    
    
end

%% Setup
[model] = feval(fh.setmodel, test);

%% Generate some data
[time, state, observ] = feval(fh.generatedata, model);

%% Create output arrays
num_to_run = length(test.algs_to_run);
pf = cell(num_to_run,1);                % Array for particle filter results
diagnostics = cell(num_to_run,1);       % Array for diagnostics

%% Run particle filters
for aa = 1:num_to_run
    alg = test.algs_to_run(aa);
    
    % Generate algorithm parameter
    [algo] = feval(fh.setalgo, test, alg);
    
    % Run it
    [pf{aa}, diagnostics{aa}] = particle_filter(display, algo, model, fh, observ, alg, state);
    
end












%% Evaluation

% Time
running_time_bs
running_time_ekf
running_time_ukf
running_time_pfp
% running_time_mhp

% Mean ESS
mn_ess_bs  = mean(ess_bs(2:end))
mn_ess_ekf = mean(ess_ekf(2:end))
mn_ess_ukf = mean(ess_ukf(2:end))
mn_ess_pfp = mean(ess_pfp(2:end))
% mn_ess_mhp = mean(ess_mhp)

% RMSE
pf_mn = [pf_bs.mn];  mn_rmse_bs =  sqrt( mean( sum((state(:,2:end) - pf_mn(:,2:end)).^2,1) , 2 ) )
pf_mn = [pf_ekf.mn]; mn_rmse_ekf =  sqrt( mean( sum((state(:,2:end) - pf_mn(:,2:end)).^2,1) , 2 ) )
pf_mn = [pf_ukf.mn]; mn_rmse_ukf =  sqrt( mean( sum((state(:,2:end) - pf_mn(:,2:end)).^2,1) , 2 ) )
pf_mn = [pf_pfp.mn]; mn_rmse_pfp =  sqrt( mean( sum((state(:,2:end) - pf_mn(:,2:end)).^2,1) , 2 ) )

% NEES
% mn_nees_bs =  
% mn_nees_ekf = 
% mn_nees_pfp = 

%% Plot graphs

if (~flags.batch) && display.plot_after
    
    
    % Individual state components
    for dd = 1:model.ds
        
        figure, hold on, plot(time, state(dd,:), 'k', 'linewidth', 2),
        
        mn_array = [pf_bs.mn]; vr_array = cat(3,pf_bs.vr);
        plot(time, mn_array(dd,:),  'b');
        plot(time, mn_array(dd,:)+2*sqrt(squeeze(vr_array(dd,dd,:))'),  ':b');
        plot(time, mn_array(dd,:)-2*sqrt(squeeze(vr_array(dd,dd,:))'),  ':b');
        
        mn_array = [pf_ekf.mn]; vr_array = cat(3,pf_ekf.vr);
        plot(time, mn_array(dd,:),  'g');
        plot(time, mn_array(dd,:)+2*sqrt(squeeze(vr_array(dd,dd,:))'),  ':g');
        plot(time, mn_array(dd,:)-2*sqrt(squeeze(vr_array(dd,dd,:))'),  ':g');
        
        mn_array = [pf_ukf.mn]; vr_array = cat(3,pf_ukf.vr);
        plot(time, mn_array(dd,:),  'c');
        plot(time, mn_array(dd,:)+2*sqrt(squeeze(vr_array(dd,dd,:))'),  ':c');
        plot(time, mn_array(dd,:)-2*sqrt(squeeze(vr_array(dd,dd,:))'),  ':c');
        
        mn_array = [pf_pfp.mn]; vr_array = cat(3,pf_pfp.vr);
        plot(time, mn_array(dd,:),  'r');
        plot(time, mn_array(dd,:)+2*sqrt(squeeze(vr_array(dd,dd,:))'),  ':r');
        plot(time, mn_array(dd,:)-2*sqrt(squeeze(vr_array(dd,dd,:))'),  ':r');
        
%         mn_array = [pf_mhp.mn]; vr_array = cat(3,pf_mhp.vr);
%         plot(time, mn_array(dd,:),  'm');
%         plot(time, mn_array(dd,:)+2*sqrt(squeeze(vr_array(dd,dd,:))'),  ':m');
%         plot(time, mn_array(dd,:)-2*sqrt(squeeze(vr_array(dd,dd,:))'),  ':m');
        
    end

%     % Plot particles for first component
%     bs_states = [pf_bs.x];
    
    
%     % Trajectory
%     figure, hold on, plot(state(1,:), state(2,:), 'k', 'linewidth', 2);
%     
%     mn_array = [pf_bs.mn]; vr_array = cat(3,pf_bs.vr);
%     plot(mn_array(1,:), mn_array(2,:),  'b');
%     
%     mn_array = [pf_ekf.mn]; vr_array = cat(3,pf_bs.vr);
%     plot(mn_array(1,:), mn_array(2,:),  'g');
%     
%     mn_array = [pf_pfp.mn]; vr_array = cat(3,pf_pfp.vr);
%     plot(mn_array(1,:), mn_array(2,:),  'r');
    
%     mn_array = [pf_mhp.mn]; vr_array = cat(3,pf_mhp.vr);
%     plot(mn_array(1,:), mn_array(2,:),  'm');
        
    figure, hold on, 
    plot(time, ess_bs, 'b'), 
    plot(time, ess_ekf, 'g'),
    plot(time, ess_ukf, 'c'),
    plot(time, ess_pfp, 'r'),
%     plot(time, ess_mhp, 'm')

    figure, hold on,
    plot(time, [pf_bs.rmse], 'b');
    plot(time, [pf_ekf.rmse], 'g');
    plot(time, [pf_ukf.rmse], 'c');
    plot(time, [pf_pfp.rmse], 'r');

    figure, hold on,
    nees = [pf_bs.nees]; plot(time, nees./(1+nees), 'b');
    nees = [pf_ekf.nees]; plot(time, nees./(1+nees), 'g');
    nees = [pf_ukf.nees]; plot(time, nees./(1+nees), 'c');
    nees = [pf_pfp.nees]; plot(time, nees./(1+nees), 'r');
    
end

% %% Movie
% 
% pf = pf_pfp;
% 
% mfig = figure; hold on;
% for kk = 1:model.K
%     
%     figure(mfig), clf, hold on
%     xlim([-25 25]), ylim([0 1])
%     
%     plot(pf(kk).state(1,:), exp(normalise_weights(pf(kk).weight)), 'xr')
%     plot(pf(kk).mn(1)*ones(1,2), [0 1], 'r');
%     
%     plot(state(kk)*ones(1,2), [0 1], 'b')
%     plot(-state(kk)*ones(1,2), [0 1], ':b')
%     
%     pause(0.5);
%     
% end

% figure, hold on, plot(sinusoidseparation_h(model,state(1:model.dsc,kk),state(model.dsc+1:end,kk)),'b'), plot(observ(:,kk),'r')
% err = zeros(1,model.K); for kk = 1:model.K, err(kk)=sum( (observ(:,kk)-sinusoidseparation_h(model,state(1:model.dsc,kk),state(model.dsc+1:end,kk))).^2 ); end
