function [ tau, A, prob ] = EKF_proposal( model, template, signal )
%EKF_PROPOSAL

% Things
R = model.y_obs_vr*eye(model.K);
y = signal';
ds = 2;
tau_a = model.tau_a;
tau_b = model.tau_b;

% Set tau and A to prior means
tau = model.tau_s + model.tau_b/(model.tau_a-1);
Amp = 1;
x = [tau; Amp];

% Linearise
t = (0:model.K-1)'/model.fs;
n = 0:model.dw-1;
grid = (t*model.fs*ones(1,model.dw)-ones(length(t),1)*n) - tau*model.fs;
H = sinc(grid);
H(isinf(grid))=0;
dH = [-Amp*model.fs*dsinc(grid)*template.m, H*template.m];
y_mod = y - Amp*H*template.m + dH*x;

% tau
mt = model.tau_s + model.tau_b/(model.tau_a-1);
Qt = 1;

% Amp
ma = model.A_mn;
Qa = model.A_vr;

m = [mt; ma];
Q = diag([Qt Qa]);


[m, Q] = ekf_update1(m, Q, y_mod, dH, R);
Q = Q+Q';

x = mvnrnd(m', Q)';
prob = loggausspdf(x, m, Q);

tau = x(1);
A = x(2);

end

function d = dsinc(x)

px = pi*x;
d = pi*(px.*cos(px)-sin(px))./(px.^2);

d(x==0)=0;

end