function model = sinusoidseparation_set_model

% Model parameters

model.K = 100;              % Number of time frames
model.M = 1;                % Number of sinusoids
model.dsc = 3*model.M;      % Number of continuous states
model.dsd = model.M+1;      % Number of discrete states
model.ds = 4*model.M+1;     % Dimension of the state
model.do = 10;              % Dimension of the observations

% Parameters
model.Ts = 0.5;
model.beta1 = 0.9;
model.beta2 = 0.5;
model.beta3 = 0.95;
model.beta4 = 0.05;
model.beta5 = 2;
model.beta6 = 0.5;
model.alpha1 = 0.5;
model.sigA = 0.1;
model.sigw = 0.1;
model.sigp = 0.01;
model.sigy = 0.01;
model.ptrans_pres = 0;
model.ptrans_noise = 0;
model.wmin = 0.5;
model.Amin = 0.5;

model.Q = blkdiag(model.sigA*eye(model.M), model.sigw*eye(model.M), model.sigp*eye(model.M));
model.R = model.sigy * eye(model.do);

% x1 distribution
model.x1_lin_mn = [ones(model.M,1); ones(model.M,1); zeros(model.M,1)];
model.x1_lin_vr = model.Q;
model.pprior_pres = 1;
model.pprior_noise = 0;

end