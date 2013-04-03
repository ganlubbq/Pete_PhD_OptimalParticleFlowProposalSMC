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
    display.plot_particle_paths = false;
    display.plot_colours = {'k', 'b', 'c', 'm', 'g'};
    
    % Set test options
    test.algs_to_run = [1,2,5];         % Vector of algorithm indexes to run
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

% This is going to depend on the model, e.g. whether its a mixed state or
% not, but here's some generic bits

rmse = cell(num_to_run,1);
nees = cell(num_to_run,1);
tnees = cell(num_to_run,1);

for aa = 1:num_to_run
    rmse{aa} = zeros(1,model.K);
    nees{aa} = zeros(1,model.K);
    tnees{aa} = zeros(1,model.K);
    for kk = 1:model.K
        rmse{aa}(kk) = sqrt(sum(diagnostics{aa}(kk).se.^2));
        if det(pf{aa}(kk).vr) > 1E-10
            nees{aa}(kk) = (diagnostics{aa}(kk).se'/pf{aa}(kk).vr)*diagnostics{aa}(kk).se;
            tnees{aa}(kk) = nees{aa}(kk)./(1+nees{aa}(kk));
        else
            nees{aa}(kk) = inf;
            tnees{aa}(kk) = 1;
        end
    end
end

%% Visual output

if display.text
    
    fprintf(1, '____________________________________________________________________\n');
    fprintf(1, 'Algorithm | Running Time (s) |  mean ESS  |  mean RMSE | mean TNEES \n');
    for aa = 1:num_to_run
        alg = test.algs_to_run(aa);
        fprintf(1, '        %u |            %5.1f |      %5.1f |      %5.1f |      %5.1f \n', alg, sum([diagnostics{aa}.rt]), mean([diagnostics{aa}.ess]), mean(rmse{aa}), mean(tnees{aa}));
    end
    fprintf(1, '____________________________________________________________________\n');
    
end

if display.plot_after
    
    close all;
    
    % ESS
    fig_ess = figure; hold on;
    for aa = 1:num_to_run
        alg = test.algs_to_run(aa);
        plot(time, [diagnostics{aa}.ess], display.plot_colours{alg});
    end
    
    % RMSE
    fig_rmse = figure; hold on;
    for aa = 1:num_to_run
        alg = test.algs_to_run(aa);
        plot(time, rmse{aa}, display.plot_colours{alg});
    end
    
    % TNEES
    fig_tnees = figure; hold on;
    for aa = 1:num_to_run
        alg = test.algs_to_run(aa);
        plot(time, tnees{aa}, display.plot_colours{alg});
    end
    
end