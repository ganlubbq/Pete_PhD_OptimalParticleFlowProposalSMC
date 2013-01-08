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
model.sigx = 10^2;
model.sigr = 1^2;
model.sigth = (30*2*pi/360)^2;
model.sigrr = 1^2;
model.alphav = 0.95;

% Matrixes
model.A = [1 0 T 0; 0 1 0 T; 0 0 model.alphav 0; 0 0 0 model.alphav];
model.Q = model.sigx * ...
    [T^3/3  0      T^2/2  0    ;
     0      T^3/3  0      T^2/2;
     T^2/2  0      T      0    ;
     0      T^2/2  0      T    ];
model.R = diag([model.sigth, model.sigr, model.sigrr]);

% x1 distribution
model.x1_mn = [-500 500 10 0]';
model.x1_vr = diag([10 10 1 1]);

end