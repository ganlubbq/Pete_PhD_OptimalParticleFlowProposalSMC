% Batch testing script for SUPF comparisons

% Add toolboxes to path
addpath('../../toolbox/user');
addpath('../../toolbox/lightspeed');
addpath('../../toolbox/ekfukf');

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

% Which model?
model_flag = 6;     % 1 = linear Gaussian
                    % 2 = nonlinear non-Gaussian benchmark
                    % 3 = heartbeat alignment
                    % 4 = tracking
                    % 5 = drone navigation
                    % 6 = parametric sine-wave heartbeat alignment
                    
% Set display options
display.text = true;
display.plot_during = false;
display.plot_after = false;
display.plot_particle_paths = false;

switch model_flag
    case 2
        test_name = 'nlng_alg_comp';
        test.algs_to_run = [1 4 6];
        test.num_filt_pts = [18500, NaN, NaN, 70, NaN, 540];
        test.flag_stochastic = false;
        
        test.STdof = Inf;
        
    case 5
        test_name = 'drone_alg_comp';
        test.algs_to_run = [1 3 4 6];
%         test.num_filt_pts = [6000 NaN 460 10 100 180];
        test.num_filt_pts = [12000, 5000, 1000, 200, NaN, 200];
        test.flag_stochastic = false; %%% Note - the IR algorithm uses stochastic updates whatever
        test.flag_intermediate_resample = false;
        test.Dscale = 0;
        
        %%% GAUSSIAN OR STUDENT-T %%%
        test.STdof = Inf;
        
    case 6
        test_name = 'sineha_alg_comp';
        test.algs_to_run = [1 3 4 6];
        test.num_filt_pts = [8000, 4000, 2000, 1000, NaN, 1000];
        test.flag_stochastic = false;
        
        test.STdof = Inf;
        
    otherwise
        error('Invalid model flag set, you chump.');
        
end

fprintf('Running test: %s', test_name);
  
% Run the script
run_pf_comparison;

% Get the numbers we want
results.rt =   rt;
results.ess =  ess;
results.rmse = rmse;

% Save
save(['batch_tests/' test_name '/batch_alg_comp_num' num2str(test_num) '.mat'], ...
    'test_num', 'test', 'model', 'algo', 'time', 'state', 'observ', 'results')

% SHOW WE'VE FINISHED
disp(['Test:' num2str(test_num) 'DONE!']);
