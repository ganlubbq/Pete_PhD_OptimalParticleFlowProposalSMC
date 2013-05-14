clup

%%% SETTTINGS %%%
filename = 'example_particle_trajectories_nlg_rs0_kk7_stoch';
%%%%%%%%%%%%%%%%%

% Load data
load([filename '.mat']);

if ~exist('state_evo','var')
    state_evo = cat(1, zeros(1,algo.N,51), state_traj);
    weight_evo = weight_traj';
end

% State evolution of first two dimensions
fig_state = figure;
hold on;
xlim([-40 40]);
ylim([-40 40]);
xlabel('x_1')
ylabel('x_2')
for ii = 1:algo.N
    plot(squeeze(state_evo(2,ii,:)), squeeze(state_evo(3,ii,:)), ':');
    plot(squeeze(state_evo(2,ii,1)), squeeze(state_evo(3,ii,1)), 'o');
    plot(squeeze(state_evo(2,ii,end)), squeeze(state_evo(3,ii,end)), 'x');
end
drawnow;
export_pdf(fig_state, [filename '_state_evo.pdf']);

% Weight evolutions
fig_weight = figure;
hold on;
xlabel('\lambda');
ylabel('log(w_{\lambda})');
for ii = 1:algo.N
    plot(lam_rng, weight_evo(:,ii), 'color', [0 rand rand]);
end
drawnow;

%% Extra plot for resampling version
fig_state_zoom = figure;
hold on;
xlim([-5 10]);
ylim([5 20]);
xlabel('x_1')
ylabel('x_2')
for ii = 1:algo.N
    plot(squeeze(state_evo(2,ii,:)), squeeze(state_evo(3,ii,:)), ':');
    plot(squeeze(state_evo(2,ii,1)), squeeze(state_evo(3,ii,1)), 'o');
    plot(squeeze(state_evo(2,ii,end)), squeeze(state_evo(3,ii,end)), 'x');
end
drawnow;
export_pdf(fig_state_zoom, [filename '_state_evo_zoom.pdf']);
