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
    
    % DEFINE RANDOM SEED
    rand_seed = 0;
    
    % Set random seed
    s = RandStream('mt19937ar', 'seed', rand_seed);
    RandStream.setDefaultStream(s);
    
    % Set flag to non-batch
    flags.batch = false;
    
    % Set function handles
    fh.set_model = @nlbenchmark_set_model;
    fh.set_algo = @nlbenchmark_set_algo;
    fh.generate_data = @nlbenchmark_generate_data;
    fh.f = @nlbenchmark_f;
    fh.h = @nlbenchmark_h;
    fh.transition = @nlbenchmark_transition;
    fh.observation = @nlbenchmark_observation;
    fh.state_prior = @nlbenchmark_stateprior;
    fh.EKFstateproposal = @nlbenchmark_EKFstateproposal;
    fh.PFstateproposal = @nlbenchmark_PFstateproposal;
    
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


%% Evaluation

% Time
running_time_bs
running_time_ekf
running_time_pfp

% Mean ESS
mn_ess_bs  = mean(ess_bs)
mn_ess_ekf = mean(ess_ekf)
mn_ess_pfp = mean(ess_pfp)

% RMSE
mn_rmse_bs =  sqrt( mean( (state - [pf_bs.mn]).^2  ) )
mn_rmse_ekf = sqrt( mean( (state - [pf_ekf.mn]).^2 ) )
mn_rmse_pfp = sqrt( mean( (state - [pf_pfp.mn]).^2 ) )

% NEES
mn_nees_bs =  mean( 1./(1 + (state - [pf_bs.mn]).^2  ./ ([pf_bs.sd].^2)) )
mn_nees_ekf = mean( 1./(1 + (state - [pf_ekf.mn]).^2 ./ ([pf_ekf.sd].^2)) )
mn_nees_pfp = mean( 1./(1 + (state - [pf_pfp.mn]).^2 ./ ([pf_pfp.sd].^2)) )

%% Plot graphs

if (~flags.batch) && display.plot_after
    
    figure, hold on, plot(time, state, 'k', 'linewidth', 2),
        plot(time, [pf_bs.mn],  'b');
        plot(time, [pf_ekf.mn], 'g');
        plot(time, [pf_pfp.mn], 'r');
        plot(time, [pf_bs.mn] +2*[pf_bs.sd],  ':b');
        plot(time, [pf_ekf.mn]+2*[pf_ekf.sd], ':g');
        plot(time, [pf_pfp.mn]+2*[pf_pfp.sd], ':r');
        plot(time, [pf_bs.mn] -2*[pf_bs.sd],  ':b');
        plot(time, [pf_ekf.mn]-2*[pf_ekf.sd], ':g');
        plot(time, [pf_pfp.mn]-2*[pf_pfp.sd], ':r');
        
    figure, hold on, plot(time, ess_bs, 'b'), plot(time, ess_ekf, 'g'), plot(time, ess_pfp, 'r')
    
end
