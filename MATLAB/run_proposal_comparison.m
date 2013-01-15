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
    model_flag = 1;     % 1 = nonlinear benchmark, 2 = target tracking
    
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
[pf_bs, ess_bs, running_time_bs] =    pf_standard(display, algo, model, fh, observ, 1);
[pf_ekf, ess_ekf, running_time_ekf] = pf_standard(display, algo, model, fh, observ, 2);
[pf_pfp, ess_pfp, running_time_pfp] = pf_standard(display, algo, model, fh, observ, 3);
% [pf_mhp, ess_mhp, running_time_mhp] = pf_standard(display, algo, model, fh, observ, 4);


%% Evaluation

% Time
running_time_bs
running_time_ekf
running_time_pfp
% running_time_mhp

% Mean ESS
mn_ess_bs  = mean(ess_bs)
mn_ess_ekf = mean(ess_ekf)
mn_ess_pfp = mean(ess_pfp)
% mn_ess_mhp = mean(ess_mhp)

% RMSE
mn_rmse_bs =  sqrt( mean( (state - [pf_bs.mn]).^2 , 2 ) )
mn_rmse_ekf = sqrt( mean( (state - [pf_ekf.mn]).^2, 2 ) )
mn_rmse_pfp = sqrt( mean( (state - [pf_pfp.mn]).^2, 2 ) )
% mn_rmse_mhp = sqrt( mean( (state - [pf_mhp.mn]).^2, 2 ) )

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
        plot(time, mn_array(dd,:)+2*sqrt(vr_array(dd,dd,:)),  ':b');
        plot(time, mn_array(dd,:)-2*sqrt(vr_array(dd,dd,:)),  ':b');
        
        mn_array = [pf_ekf.mn]; vr_array = cat(3,pf_ekf.vr);
        plot(time, mn_array(dd,:),  'g');
        plot(time, mn_array(dd,:)+2*sqrt(vr_array(dd,dd,:)),  ':g');
        plot(time, mn_array(dd,:)-2*sqrt(vr_array(dd,dd,:)),  ':g');
        
        mn_array = [pf_pfp.mn]; vr_array = cat(3,pf_pfp.vr);
        plot(time, mn_array(dd,:),  'r');
        plot(time, mn_array(dd,:)+2*sqrt(vr_array(dd,dd,:)),  ':r');
        plot(time, mn_array(dd,:)-2*sqrt(vr_array(dd,dd,:)),  ':r');
        
%         mn_array = [pf_mhp.mn]; vr_array = cat(3,pf_mhp.vr);
%         plot(time, mn_array(dd,:),  'r');
%         plot(time, mn_array(dd,:)+2*sqrt(vr_array(dd,dd,:)),  ':m');
%         plot(time, mn_array(dd,:)-2*sqrt(vr_array(dd,dd,:)),  ':m');
        
    end
    
    % Trajectory
    figure, hold on, plot(state(1,:), state(2,:), 'k', 'linewidth', 2);
    
    mn_array = [pf_bs.mn]; vr_array = cat(3,pf_bs.vr);
    plot(mn_array(1,:), mn_array(2,:),  'b');
    
    mn_array = [pf_ekf.mn]; vr_array = cat(3,pf_bs.vr);
    plot(mn_array(1,:), mn_array(2,:),  'g');
    
    mn_array = [pf_pfp.mn]; vr_array = cat(3,pf_pfp.vr);
    plot(mn_array(1,:), mn_array(2,:),  'r');
    
%     mn_array = [pf_mhp.mn]; vr_array = cat(3,pf_mhp.vr);
%     plot(mn_array(1,:), mn_array(2,:),  'm');
        
    figure, hold on, 
    plot(time, ess_bs, 'b'), 
    plot(time, ess_ekf, 'g'),
    plot(time, ess_pfp, 'r'),
%     plot(time, ess_mhp, 'm')
    
end
