function model = linearGaussian_set_model

% Model parameters

%%% SETTINGS %%%

T = 1;              % Sampling period

%%%%%%%%%%%%%%%%

% General things
model.K = 100;      % Number of time points
model.ds = 4;       % Dimension of the states
model.do = 2;       % Dimension of the observations

% Parameters
model.sigx = 10^2;
model.sigy = 0.1^2;
model.alphav = 0.9;

% Matrixes
model.A = [1 0 T 0; 0 1 0 T; 0 0 model.alphav 0; 0 0 0 model.alphav];
model.Q = model.sigx * ...
    [T^3/3  0      T^2/2  0    ;
     0      T^3/3  0      T^2/2;
     T^2/2  0      T      0    ;
     0      T^2/2  0      T    ];

model.H = [0.5 0.25 0.25 0  ;
           0   0.25 0.25 0.5];
model.R = model.sigy * ...
    [1   0.8;
     0.8 1  ];

% x1 distribution
model.x1_mn = [-50 50 1 0]';
model.x1_vr = diag([10 10 1 1]);

end