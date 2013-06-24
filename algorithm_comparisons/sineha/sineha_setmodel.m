function model = sineha_setmodel(test)

% Model parameters

% Using the parametric sinewave heartbeat alignment model with Gaussian noise.

%%%%%%%%%%%%%%%%

% General things
model.K = 10;              % Number of time points
model.do = 45;              % Dimension of the observations
model.ds = 6;               % Dimension of the state

% Transition model
model.A_shape = 10;
model.T_shape = 100;
model.tau_shape = 2;
model.omega_vr = 0.01^2;
model.phi_vr = 0.03^2;
model.B_vr = 0.1^2;

% x1 distribution
model.A1_mn = 1;
model.T1_mn = 0.2;
model.tau1_mn = 1;
model.omega1_mn = 2*pi*5;
model.phi1_mn = pi;
model.B1_mn = 0;

% Observation model
y_vr = 0.2^2;
model.R = y_vr*eye(model.do);
model.fs = 30;

end