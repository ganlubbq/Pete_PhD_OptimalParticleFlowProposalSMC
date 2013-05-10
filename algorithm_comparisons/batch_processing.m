% Script to eat folders of batch test results and make pretty tables and
% graphs

%% Algorithm comparisons

path = 'batch_tests/nls_alg_comp/';
name = 'batch_alg_comp_num';
num_mc_runs = 100;
num_tests = 6;

collated_results = repmat(struct('rt',[],'ess',[],'rmse',[]), num_mc_runs, num_tests);

for mm = 1:num_mc_runs
    
    % Open file
    load([path name num2str(mm)])
    
    % Collate
    collated_results(mm, :) = results;
    
end

% Do some averaging
average_results = repmat(struct('rt',[],'ess',[],'rmse',[]), 1, num_tests);
for tt = 1:num_tests
    
    average_results(1,tt).rt   = mean([collated_results(:,tt).rt]);
    average_results(1,tt).ess  = mean([collated_results(:,tt).ess]);
    average_results(1,tt).rmse = mean([collated_results(:,tt).rmse]);
    
end

% Table
fprintf(['%-32s & %-6s  & %-6s  & %-6s  \\\\ \n'...
         '%-32s & %2.1f & %2.1f & %2.1f \\\\ \n'...
         '%-32s & %2.1f & %2.1f & %2.1f \\\\ \n'...
         '%-32s & %2.1f & %2.1f & %2.1f \\\\ \n'...
         '%-32s & %2.1f & %2.1f & %2.1f \\\\ \n'...
         '%-32s & %2.1f & %2.1f & %2.1f \\\\ \n'...
         '%-32s & %2.1f & %2.1f & %2.1f \\\\ \n'],...
         'Algorithm','RT','ESS','RMSE',...
         'BF ($N_F=100$)',                    average_results(1).rt, average_results(1).ess, average_results(1).rmse, ...
         'BF ($N_F=185$)',                    average_results(2).rt, average_results(2).ess, average_results(2).rmse, ...
         'L-OID ($N_F=100$)',                 average_results(3).rt, average_results(3).ess, average_results(3).rmse, ...
         'SUPF ($N_F=100, \lfdiffsf=0$)',     average_results(4).rt, average_results(4).ess, average_results(4).rmse, ...
         'SUPF ($N_F=100, \lfdiffsf=0.1$)',   average_results(5).rt, average_results(5).ess, average_results(5).rmse, ...
         'SUPF ($N_F=100, \lfdiffsf=1$)',     average_results(6).rt, average_results(6).ess, average_results(6).rmse );

