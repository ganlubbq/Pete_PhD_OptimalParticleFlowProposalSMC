function model = tracking_set_model

% Model parameters

%%% SETTINGS %%%

T = 1;              % Sampling period

%%%%%%%%%%%%%%%%

% General things
model.K = 100;      % Number of time points
model.ds = 4;       % Dimension of the states
model.do = 3;       % Dimension of the observations

% Parameters
model.sigx = 0.1^2;
model.sigh = (10*2*pi/360)^2;
model.sigs = 0.1^2;

model.sigr = 0.01^2;
model.sigth = (45*2*pi/360)^2;
model.sigrr = 10^2;

model.alphav = 0.95;
model.min_speed = 0.4;

% Matrixes
% model.A = [1 0 T 0; 0 1 0 T; 0 0 model.alphav 0; 0 0 0 model.alphav];
% model.Q = model.sigx * ...
%     [T^3/3  0      T^2/2  0    ;
%      0      T^3/3  0      T^2/2;
%      T^2/2  0      T      0    ;
%      0      T^2/2  0      T    ];
model.Q = diag([model.sigx, model.sigx, model.sigh, model.sigs]);
model.R = diag([model.sigth, model.sigr, model.sigrr]);

% x1 distribution
model.x1_mn = [-50 50 0 1]';
model.x1_vr = diag([10 10 0.1 0.1]);

end