function [state, ppsl_prob] = ha_linearisedoidproposal( model, prev_state, obs, state )
%ha_linearisedoidproposal Sample and/or evaluate approximation of the OID
%for linearised around a local maximum for the heartbeat alignment model.

% If prev_state is empty, assume that this is the first step, so use the
% prior density instead of the transition density.

% Prior
if isempty(prev_state)
    A_mn = model.A1_mn;
    A_vr = model.A1_vr;
else
    A_mn = prev_state(2);
    A_vr = model.A_vr;
end

R = model.R;

% Maximise OID
% start_x = [model.tau_shift+model.tau_shape*model.tau_scale; A_mn];
start_x = [model.tau_shift+gamrnd(model.tau_shape,model.tau_scale); mvnrnd(A_mn, A_vr)];
h_of = @(x) log_oid_with_derivs(model, A_mn, A_vr, obs, R, x);
options = optimset('GradObj','on','Hessian','on','Display','notify-detailed');
% lin_x = fminunc(h_of,start_x,options);
LB = start_x - [0.05; 5];
UB = start_x + [0.05; 5];
lin_x = fmincon(h_of, start_x, [], [], [], [], LB, UB, [], options);

% figure(1), hold on
% plot([start_x(1) lin_x(1)], [start_x(2) lin_x(2)]);
% plot(lin_x(1), lin_x(2), 'o');


% Laplace Approximation
[~, ~, Hess] = log_oid_with_derivs(model, A_mn, A_vr, obs, R, lin_x);
ppsl_mn = lin_x;
ppsl_vr = inv(Hess);
ppsl_vr = (ppsl_vr+ppsl_vr')/2;
assert(isposdef(ppsl_vr));

% Sample state if not provided
if (nargin<4)||isempty(state)
    state = mvnrnd(ppsl_mn', ppsl_vr)';
end

% Calculate probability if required
if nargout>1
    ppsl_prob = loggausspdf(state, ppsl_mn, ppsl_vr);
else
    ppsl_prob = [];
end

end

function [func, grad, Hess] = log_oid_with_derivs(model, A_mn, A_vr, y, R, x)
% Negative Log-OID with gradient and Hessian

h_x = ha_h(model, x);
H = ha_obsjacobian(model, x);
dy = y - h_x;
tau = x(1);
A = x(2);

% Function itself
func_prior = - (A-A_mn)^2/(2*A_vr) - (tau-model.tau_shift)/model.tau_scale + (model.tau_shape-1)*log(tau-model.tau_shift);
if isinf(model.dfy)
    func_lhood = -(dy'/R)*dy/2;
else
    df = model.dfy;
    do = model.do;
    distp1 = (1+dy'*(R\dy)/df);
    inv1pdist = 1/distp1;
    func_lhood = -0.5*(df+do)*log(distp1);
end
func = - func_lhood - func_prior;

% Gradient
grad_prior = [ (model.tau_shape-1)/(tau-model.tau_shift) - 1/model.tau_scale; -(A-A_mn)/A_vr ];
if isinf(model.dfy)
    grad_lhood = H'*(R\dy);
else
    grad_lhood = -0.5*(df+do)*inv1pdist*( -(2/df)*H'*(R\dy) );
end
grad = - grad_lhood - grad_prior;

if nargout > 2
    
    Hess_prior = diag([ -(model.tau_shape-1)/(tau-model.tau_shift)^2, -1/A_vr ]);
    
    yR = (R\dy);
    
    tau = x(1);
    A = x(2);
    fs = model.fs;
    tmp = model.template;
    t = (0:model.do-1)'/model.fs;
    grid = (0:model.dw-1)' - tau*model.fs;
    
    % Gaussian likelihood Hessian
    Hess_lhood = - H'*(R\H);
    
    at = zeros(model.ds);         % The awkward term
    for ii = 1:model.do
        
        ds = dsinc(t(ii)*model.fs-grid);
        dds = ddsinc(t(ii)*model.fs-grid);
        
        att1 = A*fs^2*dds'*tmp;
        att2 = -fs*ds'*tmp;
        at = at + yR(ii) * [att1 att2; att2 0];
    end
    
    if isposdef(Hess_lhood+at)
        Hess_lhood = Hess_lhood + at;
    end
    
    if ~isinf(model.dfy)
        
        sf = ((df+do)/df)*inv1pdist;
        
        et = -inv1pdist*(H'/R)*dy*dy'*(R\H);
        
        if isposdef(Hess_lhood + et)
            Hess_lhood = Hess_lhood + et;
        end
        
        Hess_lhood = sf*Hess_lhood;
        
    end
    
    Hess = - Hess_lhood - Hess_prior;
    
end

end

function d = dsinc(x)

px = pi*x;
d = pi*(px.*cos(px)-sin(px))./(px.^2);

d(x==0)=0;

end

function dd = ddsinc(x)

px = pi*x;
dd = pi^2*(  (2-px.^2).*sin(px) - 2*px.*cos(px)  )./(px.^3);

dd(x==0)=-(pi^2)/3;

end

