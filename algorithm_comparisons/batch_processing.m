% Script to eat folders of batch test results and make pretty tables and
% graphs

%% Algorithm comparisons

%%% SETTINGS %%%
model_flag = 5;
num_mc_runs = 100;
studentt = false;
%%%%%%%%%%%%%%%%

alg_names = {'Bootstrap',...
             'EKF Proposal',...
             'UKF Proposal',...
             'Optimal Gaussian Proposal',...
             'Stochastic Composite Proposal with Resampling'...
             'Deterministic Composite Proposal'};

switch model_flag
    case 2
        test_name = 'nlng_alg_comp';
        algs_to_run = [1 3 4 6];
        num_filt_pts = [18500, NaN, NaN, 70, NaN, 540];
    case 5
        test_name = 'drone_alg_comp';
%         if studentt
%             test_name = [test_name '_studentt'];
%         else
% %             test_name = [test_name '_gaussian'];
%             test_name = [test_name '_with_RMstochastic'];
%         end
        algs_to_run = [1 3 4 6];
        num_filt_pts = [6000 NaN 460 10 150 180];
    case 6
        test_name = 'sineha_alg_comp';
        algs_to_run = [1 3 4 6];
        num_filt_pts = [15000, NaN, NaN, 200, NaN, 800];
    otherwise
        error('Invalid model flag set, you chump.');        
end
num_to_run = length(algs_to_run);

K = 100;
path = ['may2014_batch_tests/' test_name '/'];
name = 'batch_alg_comp_num';

rt = zeros(num_to_run,num_mc_runs);
ess = zeros(num_to_run,K,num_mc_runs);
rmse = zeros(num_to_run,K,num_mc_runs);

fails = [];
for mm = 1:num_mc_runs
    
    if exist([path name num2str(mm) '.mat'], 'file')==2
        % Open file
        load([path name num2str(mm)])
        
        for aa = 1:num_to_run
            % Collate
            rt(aa,mm) = results.rt(aa);
            ess(aa,:,mm) = results.ess{aa};
            rmse(aa,:,mm) = results.rmse{aa};
        end
    else
        fails = [fails mm];
    end
    
end

rt(:,fails) = [];
ess(:,:,fails) = [];
rmse(:,:,fails) = [];

mean_rt = mean(rt,2);
mean_ess = mean(mean(ess,3),2);
mean_rmse = mean(mean(rmse,3),2);

% Print string
fprintf('%-40s & %-4s & %-3s  & %-3s  \\\\ \n', 'Algorithm','$N_F$','ESS','RMSE')

for aa = 1:num_to_run
    fprintf('%-40s & %5u & %4.1f & %4.1f \\\\ \n', alg_names{algs_to_run(aa)}, test.num_filt_pts(algs_to_run(aa)), mean_ess(aa), mean_rmse(aa));
end

%% Draw an example plot
plot_colours = {'k', 'b', 'c', 'm', 'g', 'y', 'm'};
line_styles  = {'-', ':', ':', '-.', '--', '--'};
figure, hold on
for aa = 1:num_to_run
    plot(rmse(aa,:,4), 'color', plot_colours{algs_to_run(aa)});
end
