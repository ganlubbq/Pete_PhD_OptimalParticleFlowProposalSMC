function [state, ppsl_prob] = tracking_linearisedoidproposal( model, prev_state, obs, state )
%tracking_linearisedoidproposal Sample and/or evaluate approximation of the
% OID linearised around a local maximum for the tracking model.

% If prev_state is empty, assume that this is the first step, so use the
% prior density instead of the transition density.

% Prior
if isempty(prev_state)
    prior_mn = model.m1;
    prior_vr = model.P1;
else
    prior_mn = model.A*prev_state;
    prior_vr = model.Q;
end

% Sample a mixing value and scale the covariance
if ~isinf(model.dfx)
    xi = chi2rnd(model.dfx);
else
    xi = 1;
end
prior_vr = prior_vr / xi;

% Maximise OID
R = model.R;
h_of = @(x) log_oid_with_derivs(model, prior_mn, prior_vr, obs, R, x);
options = optimset('GradObj','on','Hessian','on','Display','notify-detailed');
options = optimset('GradObj','on','Display','notify-detailed');
start_x = prior_mn+2*rand(model.ds,1)-1;
lin_x = fminunc(h_of,start_x,options);

% Laplace Approximation
[~, ~, Hess] = log_oid_with_derivs(model, prior_mn, prior_vr, obs, R, lin_x);
ppsl_mn = lin_x;
ppsl_vr = inv(Hess);
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

function [func, grad, Hess] = log_oid_with_derivs(model, m, Q, y, R, x)
% Negative Log-OID with gradient and Hessian

h_x = tracking_h(model, x);
H = tracking_obsjacobian(model, x);

dy = y - h_x;
dx = x - m;
func = ( (dy'/R)*dy + (dx'/Q)*dx )/2;
grad = - H'*(R\dy) + Q\dx;

if nargout > 2
    
    T = tracking_obssecondderivtensor(model, x);
    yR = (R\dy);
    
    Hess = (H'/R)*H + inv(Q);
    
    at = zeros(model.ds);
    for ii = 1:model.do
        at = at + squeeze(T(ii,:,:))*yR(ii);
    end
    
    if isposdef(Hess - at)
        Hess = Hess - at;
    end
    
end

end

