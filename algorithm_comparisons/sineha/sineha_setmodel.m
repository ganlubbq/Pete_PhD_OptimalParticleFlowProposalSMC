function model = sineha_setmodel(test)

% Model parameters

% Using the parametric sinewave heartbeat alignment model with Gaussian noise.

%%%%%%%%%%%%%%%%

% General things
model.K = 100;              % Number of time points
model.do = 50;              % Dimension of the observations
model.ds = 6;               % Dimension of the state

% Transition model
model.A_shape = 10; model.A_scale = 0.05; model.A_shift = 0.5;
model.T_vol = 0.01;
model.tau_shape = 20; model.tau_scale = 0.5/model.tau_shape;
model.omega_vr = 0.5^2;
model.phi_vr = 0.5^2;
model.B_vr = 0.5^2;

% x1 distribution
model.T1_mn = 0.2;
model.omega1_mn = 2*pi*5;
model.phi1_mn = pi;
model.B1_mn = 0;

% Observation model
y_vr = 0.2^2;
model.R = y_vr*eye(model.do);
model.fs = 30;

end