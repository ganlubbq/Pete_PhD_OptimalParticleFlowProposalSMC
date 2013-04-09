function [ v, A ] = flow_velocity( model, lam, x, template, signal, D )
%FLOW_VELOCITY 

tau = x(1);
Amp = x(2);

% x(2) = [];

% Things
R = model.y_obs_vr*eye(model.K);
y = signal';
ds = 2;
tau_a = model.tau_a;
tau_b = model.tau_b;
tau_s = model.tau_s;

% Linearise
t = (0:model.K-1)'/model.fs;
n = 0:model.dw-1;
grid = (t*model.fs*ones(1,model.dw)-ones(length(t),1)*n) - tau*model.fs;
H = sinc(grid);
H(isinf(grid))=0;
dH = [-Amp*model.fs*dsinc(grid)*template.m, H*template.m];
% dH = -Amp*model.fs*dsinc(grid)*template.m;
y_mod = y - Amp*H*template.m + dH*x;
% R = R + Amp^2*H*template.P*H';

% Use matching to find equivalent Gaussian

% tau
p0 = invgampdf(tau-tau_s, model.tau_a, model.tau_b);
d0 = ((tau_b^tau_a)/gamma(tau_a))*(tau-tau_s)^(-(tau_a+1)-2)*exp(-tau_b/(tau-tau_s))*(tau_b-(tau_a+1)*(tau-tau_s));
Del = -d0/p0;
Qt = numerical_lambertw(Del^2/(2*pi*p0^2))/Del^2;
mt = tau - Del*Qt;

if isnan(Qt)||isinf(Qt)
    mt = model.tau_s + model.tau_b/(model.tau_a-1);
    Qt = 1;
end

% Amp
ma = model.A_mn;
Qa = model.A_vr;

m = [mt; ma];
Q = diag([Qt Qa]);
% m = mt;
% Q = Qt;
% ds = 1;

% Find particle velocity using Gaussian approximation
Sig_inv = inv(Q)+lam*dH'*(R\dH);
A = -0.5*Q*dH'*((R+lam*dH*Q*dH')\dH);
AD = A - D*Sig_inv;
b = (eye(ds)+2*lam*A)*((eye(ds)+lam*A)*Q*dH'*(R\y_mod)+A*m);
bD = b + D*(Q\m + lam*dH'*(R\y_mod));



v = AD * x + bD;


% v = [v; 0];


end

function d = dsinc(x)

assert(all(x(:)~=0))

px = pi*x;
d = pi*(px.*cos(px)-sin(px))./(px.^2);



end

