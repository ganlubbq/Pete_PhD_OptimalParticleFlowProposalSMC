function model = nlbenchmark_set_model

% Model parameters

model.K = 100;      % Number of time points
model.ds = 1;       % Dimension of the states
model.do = 1;       % Dimension of the observations

% Parameters
model.beta1 = 0.5;
model.beta2 = 25;
model.beta3 = 8;
model.alpha1 = 0.05;
model.alpha2 = 2;
model.sigx = 10;
model.sigy = 1;

% x1 distribution
model.x1_mn = 0;
model.x1_vr = 5;

end