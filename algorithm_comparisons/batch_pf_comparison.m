% Batch testing script for SUPF comparisons

% Add toolboxes to path
addpath('../toolbox/user');
addpath('../toolbox/lightspeed');
addpath('../toolbox/ekfukf');

% Clean up
clup

% Get environment variable specifying test number
test_num = str2double(getenv('SGE_TASK_ID'));
% test_num = 0;

% Set flags
test.flag_batch = true;

% DEFINE RANDOM SEED
rand_seed = test_num;

%%% SETTINGS %%%

% TEST NAME
test_name = 'nlg_alg_comp';

% Which model?
model_flag = 2;     % 1 = linear Gaussian
                    % 2 = nonlinear non-Gaussian benchmark
                    % 3 = heartbeat alignment

% Gaussian or Student-t
if model_flag == 2
    test.STdof = Inf;
end
                    
% Set display options
display.text = false;
display.plot_during = false;
display.plot_after = false;
display.plot_particle_paths = false;

% How many tests
num_tests = 7;


                                            % Algorithm Numbers
                                            % 1 = bootstrap
                                            % 2 = EKF proposal
                                            % 3 = UKF proposal
                                            % 4 = linearised OID proposal
                                            % 5 = SUPF

% Loop through tests
for tt = 1:num_tests
    
    % Set generic things
    test.num_filt_pts = 100*ones(1,5);
    
    if model_flag == 1
        % Set test-specific things - LINEAR GAUSSIAN
        switch tt
            case 1
                test.algs_to_run = [1];
                test.num_filt_pts(1) = 100;
            case 2
                test.algs_to_run = [1];
                test.num_filt_pts(1) = 185;
            case 3
                test.algs_to_run = [4];
                test.num_filt_pts(4) = 100;
            case 4
                test.algs_to_run = [5];
                test.num_filt_pts(5) = 100;
                test.flag_stochastic = false;
            case 5
                test.algs_to_run = [5];
                test.num_filt_pts(5) = 100;
                test.flag_stochastic = true;
                test.Dscale = 0.1;
            case 6
                test.algs_to_run = [5];
                test.num_filt_pts(5) = 100;
                test.flag_stochastic = true;
                test.Dscale = 1;
            otherwise
                error('No settings for that test');
        end
    elseif model_flag == 2
        % Set test-specific things - NONLINEAR GAUSSIAN
        switch tt
            case 1
                test.algs_to_run = [1];
                test.num_filt_pts(1) = 100;
                test.flag_intermediate_resample = false;
            case 2
                test.algs_to_run = [1];
                test.num_filt_pts(1) = 20000;
                test.flag_intermediate_resample = false;
            case 3
                test.algs_to_run = [4];
                test.num_filt_pts(4) = 100;
                test.flag_intermediate_resample = false;
            case 4
                test.algs_to_run = [5];
                test.num_filt_pts(5) = 100;
                test.flag_stochastic = false;
                test.flag_intermediate_resample = false;
            case 5
                test.algs_to_run = [5];
                test.num_filt_pts(5) = 100;
                test.flag_stochastic = true;
                test.flag_intermediate_resample = false;
                test.Dscale = 0.01;
            case 6
                test.algs_to_run = [5];
                test.num_filt_pts(5) = 100;
                test.flag_stochastic = true;
                test.flag_intermediate_resample = false;
                test.Dscale = 0.1;
            case 7
                test.algs_to_run = [5];
                test.num_filt_pts(5) = 100;
                test.flag_stochastic = true;
                test.flag_intermediate_resample = true;
                test.Dscale = 0.01;
        end
    end
    
    % Run the script
    run_pf_comparison;
    
    % Get the numbers we want
    results(tt).rt =   sum([diagnostics{aa}.rt]);
    results(tt).ess =  mean([diagnostics{aa}(2:end).ess]);
    results(tt).rmse = mean(rmse{aa}(2:end));
    
end

% Save
save(['batch_tests/' test_name '/batch_alg_comp_num' num2str(test_num) '.mat'], 'results')

% SHOW WE'VE FINISHED
disp(['Test:' num2str(test_num) 'DONE!']);
