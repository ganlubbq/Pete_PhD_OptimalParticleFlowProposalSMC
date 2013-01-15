function model = nlbenchmark_set_model

% Model parameters

model.K = 100;      % Number of time points
model.ds = 10;       % Dimension of the states
model.do = model.ds;%/2;       % Dimension of the observations

% Parameters
model.beta1 = 0.5;
model.beta2 = 25;
model.beta3 = 8;
model.alpha1 = 0.05;
model.alpha2 = 2;
model.sigx = 10;
model.sigy = 1;

model.Q = model.sigx * eye(model.ds);
model.R = model.sigy * eye(model.do);

model.Hlin = eye(model.do);
% model.Hlin = zeros(model.do, model.ds);
% for ii = 1:model.do
%     model.Hlin(ii, 2*ii-1:2*ii) = 1/2;
% end

% x1 distribution
model.x1_mn = zeros(model.ds,1);
model.x1_vr = eye(model.ds);

end