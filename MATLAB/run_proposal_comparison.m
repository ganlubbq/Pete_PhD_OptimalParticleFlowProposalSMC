% Base script for running particle filter proposal comparison

% Structures
% display - display options while running
% algo - algorithm parameters
% model - model parameters

%% Set Up

if ~exist('flags.batch', 'var') || (~flags.batch)
    
    clup
    dbstop if error
    % dbstop if warning
    
    %%% SETTINGS %%%
    
    % DEFINE RANDOM SEED
    rand_seed = 0;
    
    % Which model?
    model_flag = 4;     % 1 = nonlinear benchmark, 2 = target tracking
    
    %%%%%%%%%%%%%%%%
    
    % Set random seed
    s = RandStream('mt19937ar', 'seed', rand_seed);
    RandStream.setDefaultStream(s);
    
    % Set flag to non-batch
    flags.batch = false;
    
    % Set function handles
    if model_flag == 1
        fh.set_model = @nlbenchmark_set_model;
        fh.set_algo = @nlbenchmark_set_algo;
        fh.generate_data = @nlbenchmark_generate_data;
        fh.transition = @nlbenchmark_transition;
        fh.observation = @nlbenchmark_observation;
        fh.stateprior = @nlbenchmark_stateprior;
        fh.EKFstateproposal = @nlbenchmark_EKFstateproposal;
        fh.UKFstateproposal = @nlbenchmark_UKFstateproposal;
        fh.PFstateproposal = @nlbenchmark_PFstateproposal;
        fh.MHstateproposal = @nlbenchmark_MHstateproposal;
    elseif model_flag == 2
        fh.set_model = @tracking_set_model;
        fh.set_algo = @tracking_set_algo;
        fh.generate_data = @tracking_generate_data;
        fh.transition = @tracking_transition;
        fh.observation = @tracking_observation;
        fh.stateprior = @tracking_stateprior;
        fh.EKFstateproposal = @tracking_EKFstateproposal;
        fh.PFstateproposal = @tracking_PFstateproposal;
        fh.MHstateproposal = @tracking_MHstateproposal;
    elseif model_flag == 3
        fh.set_model = @linearGaussian_set_model;
        fh.set_algo = @linearGaussian_set_algo;
        fh.generate_data = @linearGaussian_generate_data;
        fh.transition = @linearGaussian_transition;
        fh.observation = @linearGaussian_observation;
        fh.stateprior = @linearGaussian_stateprior;
        fh.EKFstateproposal = @linearGaussian_EKFstateproposal;
        fh.PFstateproposal = @linearGaussian_PFstateproposal;
        fh.MHstateproposal = @linearGaussian_MHstateproposal;
    elseif model_flag == 4
        fh.set_model = @sinusoidseparation_set_model;
        fh.set_algo = @sinusoidseparation_set_algo;
        fh.generate_data = @sinusoidseparation_generate_data;
        fh.transition = @sinusoidseparation_transition;
        fh.observation = @sinusoidseparation_observation;
        fh.stateprior = @sinusoidseparation_stateprior;
        fh.EKFstateproposal = @sinusoidseparation_EKFstateproposal;
        fh.UKFstateproposal = @sinusoidseparation_UKFstateproposal;
        fh.PFstateproposal = @sinusoidseparation_PFstateproposal;
    end
    
    % Set model and algorithm parameters
    model = feval(fh.set_model);
    algo = feval(fh.set_algo);
    
    % Set display options
    display.text = true;
    display.plot_during = false;
    if display.plot_during
        display.h_pf(1) = figure;
        display.h_pf(2) = figure;
    end
    display.plot_after = true;
    
    
end

%% Generate some data
[time, state, observ] = feval(fh.generate_data, model);

%% Run particle filters
[pf_bs, ess_bs, running_time_bs] =    pf_standard(display, algo, model, fh, observ, 1, state);
% [pf_ekf, ess_ekf, running_time_ekf] = pf_standard(display, algo, model, fh, observ, 2, state);
[pf_ukf, ess_ukf, running_time_ukf] = pf_standard(display, algo, model, fh, observ, 3, state);
[pf_pfp, ess_pfp, running_time_pfp] = pf_standard(display, algo, model, fh, observ, 4, state);
% [pf_mhp, ess_mhp, running_time_mhp] = pf_standard(display, algo, model, fh, observ, 5, state);


%% Evaluation

% Time
running_time_bs
% running_time_ekf
running_time_ukf
running_time_pfp
% running_time_mhp

% Mean ESS
mn_ess_bs  = mean(ess_bs(2:end))
% mn_ess_ekf = mean(ess_ekf(2:end))
mn_ess_ukf = mean(ess_ukf(2:end))
mn_ess_pfp = mean(ess_pfp(2:end))
% mn_ess_mhp = mean(ess_mhp)

% RMSE
pf_mn = [pf_bs.mn];  mn_rmse_bs =  sqrt( mean( sum((state(:,2:end) - pf_mn(:,2:end)).^2,1) , 2 ) )
% pf_mn = [pf_ekf.mn]; mn_rmse_ekf =  sqrt( mean( sum((state(:,2:end) - pf_mn(:,2:end)).^2,1) , 2 ) )
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
        
%         mn_array = [pf_ekf.mn]; vr_array = cat(3,pf_ekf.vr);
%         plot(time, mn_array(dd,:),  'g');
%         plot(time, mn_array(dd,:)+2*sqrt(squeeze(vr_array(dd,dd,:))'),  ':g');
%         plot(time, mn_array(dd,:)-2*sqrt(squeeze(vr_array(dd,dd,:))'),  ':g');
        
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
%     plot(time, ess_ekf, 'g'),
    plot(time, ess_ukf, 'c'),
    plot(time, ess_pfp, 'r'),
%     plot(time, ess_mhp, 'm')

    figure, hold on,
    plot(time, [pf_bs.rmse], 'b');
%     plot(time, [pf_ekf.rmse], 'g');
    plot(time, [pf_ukf.rmse], 'c');
    plot(time, [pf_pfp.rmse], 'r');

    figure, hold on,
    nees = [pf_bs.nees]; plot(time, nees./(1+nees), 'b');
%     nees = [pf_ekf.nees]; plot(time, nees./(1+nees), 'g');
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
