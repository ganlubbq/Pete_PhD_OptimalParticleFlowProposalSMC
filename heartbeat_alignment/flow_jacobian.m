function [ A ] = flow_jacobian( model, lam, x, template )
%FLOW_JACOBIAN

% Things
R = model.y_obs_vr*eye(model.K);
tau_a = model.tau_a;
tau_b = model.tau_b;

% Linearise
t = (0:model.K-1)'/model.fs;
n = 0:model.dw-1;
grid = (t*model.fs*ones(1,model.dw)-ones(length(t),1)*n) - tau*model.fs;
dH = -model.fs*dsinc(grid)*template.m;

% Use matching to find equivalent Gaussian
p0 = invgampdf(tau-model.tau_s, model.tau_a, model.tau_b);
d0 = ((tau_b^tau_a)/gamma(tau_a))*tau^(-(tau_a+1)-2)*exp(-tau_b/tau)*(tau_b-(tau_a+1)*tau);
Del = -d0/p0;
Q = lambertw_ex(Del^2/(2*pi*p0^2))/Del^2;

% Find particle velocity using Gaussian approximation
A = -0.5*Q*dH'*((R+lam*dH*Q*dH')\dH);

end

function d = dsinc(x)

assert(all(x(:)~=0))

px = pi*x;
d = pi*(px.*cos(px)-sin(px))./(px.^2);



end

