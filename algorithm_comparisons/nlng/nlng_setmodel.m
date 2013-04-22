function model = nlng_setmodel(test)

% Model parameters

% Using the a multi-dimensional generalisation of the common nonlinear
% benckmark model with additive t (Cauchy) noise.

%%%%%%%%%%%%%%%%

% General things
model.K = 100;              % Number of time points
dsc = 10;                    % Dimension of the CONTINUOUS states
model.do = dsc/2;           % Dimension of the observations
model.ds = dsc + 1;         % Dimension of the state

% Parameters
sigx = 100;
sigy = 1;

% Transition
model.beta1 = 0.5;
model.beta2 = 25;
model.beta3 = 8;

% Observation
model.alpha1 = 0.05;
model.alpha2 = 2;
% model.Hlin = [1 1];
% model.Hlin = [1 1 1 1; 1 1 0 0; 0 0 1 1];
% model.Hlin = eye(dsc);
model.Hlin = zeros(model.do, dsc);
for ii = 1:model.do
    model.Hlin(ii, 2*ii-1:2*ii) = 1;
end

% Noises
model.dfy = 3; % Number of degrees of freedom for t-distributed noise. 1 = Cauchy, inf = Normal
model.Q = sigx * eye(dsc);
model.R = sigy * eye(model.do);     % R is a spread matrix, rather than a covariance

% x1 distribution
model.m1 = zeros(dsc,1);
model.P1 = eye(dsc);

end