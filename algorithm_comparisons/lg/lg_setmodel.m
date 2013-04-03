function model = lg_setmodel(test)

% Model parameters

% Using the 2D NCV tracking model for the transition model and and some 
% random linear combinations for the observation model.

%%%%%%%%%%%%%%%%

% General things
model.K = 100;      % Number of time points
model.ds = 4;       % Dimension of the states
model.do = 2;       % Dimension of the observations

% Parameters
sigx = 10^2;
sigy = 0.1^2;
alphav = 0.9;

% Matrixes
T = 1;              % Sampling period
model.A = [1 0 T 0; 0 1 0 T; 0 0 alphav 0; 0 0 0 alphav];
model.Q = sigx * ...
    [T^3/3  0      T^2/2  0    ;
     0      T^3/3  0      T^2/2;
     T^2/2  0      T      0    ;
     0      T^2/2  0      T    ];

model.H = [0.5 0.25 0.25 0  ;
           0   0.25 0.25 0.5];
model.R = sigy * ...
    [1   0.8;
     0.8 1  ];

% x1 distribution
model.m1 = [-50 50 1 0]';
model.P1 = diag([10 10 1 1]);

end