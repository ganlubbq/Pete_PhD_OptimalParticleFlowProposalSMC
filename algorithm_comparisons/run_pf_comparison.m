% Base script for running particle filter comparisons for SUPF paper

% Structures
% display - parameters for determining MATLAB output
% test - test parameters
% algo - algorithm parameters
% model - model parameters

%% Preliminaries

if ~exist('test', 'var') || ~isfield(test,'flag_batch') || (~test.flag_batch)
    
    clup
    dbstop if error
%     dbstop if warning

    % Set flag to non-batch
    test.flag_batch = false;
    
    %%% SETTINGS %%%
    
    % DEFINE RANDOM SEED
    rand_seed = 0;
    
    % Which model?
    model_flag = 5;     % 1 = linear Gaussian
                        % 2 = nonlinear non-Gaussian benchmark
                        % 3 = heartbeat alignment
                        % 4 = tracking
                        % 5 = drone navigation
                        % 6 = parametric sine-wave heartbeat alignment
    
    % Set display options
    display.text = true;
    display.plot_during = false;    
    display.plot_after = true;
    display.plot_particle_paths = false;
    display.plot_colours = {'k', 'b', 'c', 'm', 'g', 'g', 'm', 'g'};
    if display.plot_during
        display.h_pf(1) = figure;
        display.h_pf(2) = figure;
    end
    if display.plot_particle_paths
        display.h_ppp(1) = figure;
        display.h_ppp(2) = figure;
        display.h_ppp(3) = figure;
        display.h_ppp(4) = figure;
        display.h_ppp(5) = figure;
        display.h_ppp(6) = figure;
    end
    
    % Select algorithms to run
    test.algs_to_run = [6];     % Vector of algorithm indexes to run
                                    % 1 = bootstrap
                                    % 2 = EKF proposal
                                    % 3 = UKF proposal
                                    % 4 = linearised OID proposal
                                    % 5 = SUPF
                                    % 6 = SUPF by particle
                                    % 
                                    % 8 = MCMC proposal
    
	% Set number of particles for each algorithm
%     test.num_filt_pts = 100*ones(1,8);
    test.num_filt_pts = [12000, 5000, 1000, 200, NaN, 200];          % Time normalised for model 5
%     test.num_filt_pts = [8000, 4000, 2000, 1000, NaN, 1000];          % Time normalised for model 6

%     test.num_filt_pts = [185, 100, 100, 100, 100];          % Time normalised for model 1
%     test.num_filt_pts = [18500, NaN, NaN, 70, NaN, 540];       % Time normalised for model 2 with Gaussian densities
%     test.num_filt_pts = [2000, NaN, NaN, 50, NaN, 100];       % Time normalised for model 2 with Poisson observations
%     test.num_filt_pts = [2500, 1000, 1000, 50, 100, 100];       % Time normalised for model 3
%     test.num_filt_pts = [20000 12000 3500 10 100];               % Time normalised for model 4
%     test.num_filt_pts = [6000 NaN 460 10 100 180];               % Time normalised for model 5
%     test.num_filt_pts = [15000, NaN, NaN, 200, NaN, 800, 1000];       % Time normalised for model 6

    % Model settings
    test.STdof = Inf;

    % SUPF settings
    test.flag_stochastic = true;
    test.flag_intermediate_resample = false;
    test.Dscale = 0.3;

end



%% Setup

fprintf('Running tests with model flag %u.\n', model_flag);
fprintf('   Random seed: %u.\n', rand_seed);

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
    fh.smoothupdatebyparticle = @nlng_smoothupdatebyparticle;
%     fh.smoothupdatebyparticle = @nlng_smoothupdatebyparticle_scalemix;

    fh.observation = @nlng_poisson_observation;
    fh.generatedata = @nlng_poisson_generatedata;
    fh.linearisedoidproposal = @nlng_poisson_linearisedoidproposal;
    fh.smoothupdatebyparticle = @nlng_poisson_smoothupdatebyparticle;
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
%     fh.smoothupdatebyparticle = @ha_smoothupdatebyparticle;
    fh.smoothupdatebyparticle = @ha_modifiedsmoothupdatebyparticle;
    fh.mcmcproposal = @ha_mcmcproposal;
elseif model_flag == 4
    addpath('tracking');
    fh.setmodel = @tracking_setmodel;
    fh.setalgo = @tracking_setalgo;
    fh.generatedata = @tracking_generatedata;
    fh.transition = @tracking_transition;
    fh.observation = @tracking_observation;
    fh.stateprior = @tracking_stateprior;
    fh.ekfproposal = @tracking_ekfproposal;
    fh.ukfproposal = @tracking_ukfproposal;
    fh.linearisedoidproposal = @tracking_linearisedoidproposal;
    fh.smoothupdate = @tracking_smoothupdate;
elseif model_flag == 5
    addpath('drone');
    fh.setmodel = @drone_setmodel;
    fh.setalgo = @drone_setalgo;
    fh.generatedata = @drone_generatedata;
    fh.transition = @drone_transition;
    fh.observation = @drone_observation;
    fh.stateprior = @drone_stateprior;
    fh.ekfproposal = @drone_ekfproposal;
    fh.ukfproposal = @drone_ukfproposal;
    fh.linearisedoidproposal = @drone_linearisedoidproposal;
%     fh.smoothupdate = @drone_smoothupdatewithRM;
%     fh.smoothupdate = @drone_smoothupdatewithIR;
    fh.smoothupdate = @drone_smoothupdate;
    fh.smoothupdatebyparticle = @drone_smoothupdatebyparticle;
%     fh.smoothupdatebyparticle = @drone_modifiedsmoothupdatebyparticle;
    fh.annealedupdate = @drone_annealedupdate;
elseif model_flag == 6
    addpath('sineha');
    fh.setmodel = @sineha_setmodel;
    fh.setalgo = @sineha_setalgo;
    fh.generatedata = @sineha_generatedata;
    fh.transition = @sineha_transition;
    fh.observation = @sineha_observation;
    fh.stateprior = @sineha_stateprior;
    fh.ekfproposal = @sineha_ekfproposal;
    fh.ukfproposal = @sineha_ukfproposal;
    fh.linearisedoidproposal = @sineha_linearisedoidproposal;
    fh.smoothupdate = @sineha_smoothupdate;
    fh.smoothupdatebyparticle = @sineha_smoothupdatebyparticle;
%     fh.smoothupdatebyparticle = @sineha_smoothupdatebyparticle;
    fh.smoothupdatebyparticle = @sineha_modifiedsmoothupdatebyparticle;
end

% Set random seed
rng(rand_seed);

% Set model parameters
[model] = feval(fh.setmodel, test);

%% Data simulation

% Set random seed
rng(rand_seed);

% Generate data
[time, state, observ] = feval(fh.generatedata, model);

%% Create output arrays
num_to_run = length(test.algs_to_run);
pf = cell(num_to_run,1);                % Array for particle filter results
diagnostics = cell(num_to_run,1);       % Array for diagnostics

%% Run particle filters
for aa = 1:num_to_run
    alg = test.algs_to_run(aa);
    
    % Reset random seed
    rng(rand_seed+1);
    
    % Generate algorithm parameter
    [algo] = feval(fh.setalgo, test, model, alg);
    
    % Run it
    [pf{aa}, diagnostics{aa}] = particle_filter(display, algo, model, fh, observ, alg, state);
    
end

%% Evaluation

% This is going to depend on the model, e.g. whether its a mixed state or
% not, but here's some generic bits

% sse = cell(num_to_run,1);
rmse = cell(num_to_run,1);
nees = cell(num_to_run,1);
tnees = cell(num_to_run,1);
ess = cell(num_to_run,1);
rt = zeros(num_to_run,1);

for aa = 1:num_to_run
    
    rt(aa) = diagnostics{aa}.rt;
    
%     sse{aa} = zeros(model.ds,model.K);
    rmse{aa} = zeros(1,model.K);
    nees{aa} = zeros(1,model.K);
    tnees{aa} = zeros(1,model.K);
    ess{aa} = zeros(1,model.K);
    for kk = 1:model.K
        
        ess{aa}(kk) = diagnostics{aa}(kk).ess;
        
        switch model_flag
            case{1 3 4 5 6}
                se = diagnostics{aa}(kk).se;
                vr = pf{aa}(kk).vr;
            case{2}
                se = diagnostics{aa}(kk).se(2:end);
                vr = pf{aa}(kk).vr(2:end, 2:end);
        end
        
%         sse{aa}(:,kk) = se;
        rmse{aa}(kk) = sqrt(sum(se.^2));   %abs(se(1));%
        if det(vr) > 1E-12
            nees{aa}(kk) = (se'/vr)*se;
            tnees{aa}(kk) = nees{aa}(kk)./(1+nees{aa}(kk));
        else
            nees{aa}(kk) = inf;
            tnees{aa}(kk) = 1;
        end
    end
end

%% Visual output

if ~test.flag_batch
    
    if display.text
        
        fprintf(1, '____________________________________________________________________\n');
        fprintf(1, 'Algorithm | Running Time (s) |  mean ESS  |  mean RMSE | mean TNEES \n');
        for aa = 1:num_to_run
            alg = test.algs_to_run(aa);
            fprintf(1, '        %u |            %5.1f |      %5.1f |      %5.3f |      %5.3f \n', alg, sum([diagnostics{aa}.rt]), mean([diagnostics{aa}(2:end).ess]), mean(rmse{aa}(2:end)), mean(tnees{aa}(2:end)));
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
        
%         % State errors
%         for dd = 1:model.ds
%             figure; hold on;
%             for aa = 1:num_to_run
%                 alg = test.algs_to_run(aa);
%                 plot(time, sse{aa}(dd,:), display.plot_colours{alg});
%             end
%         end
        
    end
    
    if (model_flag == 4) || (model_flag == 5)
        
        fig_traj = figure; hold on;
        plot3(state(1,:), state(2,:), state(3,:), ':k')
        plot3(state(1,1), state(2,1), state(3,1), 'ok')
        
        for aa = 1:num_to_run
            alg = test.algs_to_run(aa);
            mmse_est = [pf{aa}.mn];
            plot3(mmse_est(1,:), mmse_est(2,:), mmse_est(3,:), display.plot_colours{alg})
            plot3(mmse_est(1,1), mmse_est(2,1), mmse_est(3,1), 'o', 'color', display.plot_colours{alg})
        end
        
    end
    
    if (model_flag == 6)
       
        for dd=1:model.ds
            figure; hold on;
            plot(state(dd,:), ':k');
            for aa = 1:num_to_run
                alg = test.algs_to_run(aa);
                mmse_est = [pf{aa}.mn];
                plot(mmse_est(dd,:), 'color', display.plot_colours{alg});
            end
        
        end
        
    end
    
end