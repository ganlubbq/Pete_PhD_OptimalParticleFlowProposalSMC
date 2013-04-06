function model = ha_setmodel(test)

% Model parameters

% Using the heartbeat alignment model with t noise.

%%%%%%%%%%%%%%%%

% General things
model.K = 20;              % Number of time points
model.do = 30;              % Dimension of the observations
model.ds = 2;               % Dimension of the state

% Heartbeat template
load('template_beat.mat');
model.template = template'; clear template;
model.dw = length(model.template);

% Transition model
model.A_vr = 0.1;
model.tau_shape = 2;
model.tau_scale = 0.1;
model.tau_shift = 0;

% x1 distribution
model.A1_mn = 1;
model.A1_vr = 1;

% Observation model
y_vr = 0.2^2;1;
model.R = y_vr*eye(model.do);
model.fs = 30;
model.dfy = Inf;1;

end