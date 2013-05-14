function [state, ppsl_prob] = nlng_linearisedoidproposal( model, prev_state, obs, state )
%nlng_linearisedoidproposal Sample and/or evaluate approximation of the OID
%for linearised around a local maximum for the nonlinear non-Gaussian
%benchmark model.

% If prev_state is empty, assume that this is the first step, so use the
% prior density instead of the transition density.

% Unpack the state
if ~isempty(prev_state)
    prev_kk = prev_state(1);
    prev_x = prev_state(2:end);
else
    prev_kk = 0;
end

% Prior
if isempty(prev_state)
    prior_mn = model.m1;
    prior_vr = model.P1;
else
    prior_mn = nlng_f(model, prev_kk, prev_x);
    prior_vr = model.Q;
end

% Sample a mixing value and scale the covariance
if ~isinf(model.dfy)
    xi = chi2rnd(model.dfy);
else
    xi = 1;
end
R = model.R / xi;

% Maximise OID
h_of = @(x) log_oid_with_derivs(model, prior_mn, prior_vr, obs, R, x);
options = optimset('GradObj','on','Hessian','on','Display','notify-detailed');
options = optimset('GradObj','on','Display','notify-detailed');
start_x = prior_mn+2*rand(model.ds-1,1)-1;
lin_x = fminunc(h_of,start_x,options);

% Laplace Approximation
[~, ~, Hess] = log_oid_with_derivs(model, prior_mn, prior_vr, obs, R, lin_x);
ppsl_mn = lin_x;
ppsl_vr = inv(Hess);
assert(isposdef(ppsl_vr));

% Sample state if not provided
if (nargin<4)||isempty(state)
    kk = prev_kk + 1;
    x = mvnrnd(ppsl_mn', ppsl_vr)';
    state = [kk; x];
else
    kk = state(1);
    x = state(2:end);
    assert(kk==prev_kk+1);
end

% Calculate probability if required
if nargout>1
    ppsl_prob = loggausspdf(x, ppsl_mn, ppsl_vr);
else
    ppsl_prob = [];
end

end

function [func, grad, Hess] = log_oid_with_derivs(model, m, Q, y, R, x)
% Negative Log-OID with gradient and Hessian

h_x = nlng_h(model, x);
H = nlng_obsjacobian(model, x);
dy = y - h_x;
dx = x - m;

func = ( (dy'/R)*dy + (dx'/Q)*dx )/2;
grad = - H'*(R\dy) + Q\dx;

if nargout > 2
    
    Hess = (H'/R)*H + inv(Q);
    
    yR = (R\dy);
    
    % at = 0;
    
    ds = model.ds-1;
    nl = model.alpha1 * model.alpha2 * (model.alpha2-1) * (x.^2).^(model.alpha2/2 - 1);
    nl(isnan(nl)) = 0;
    at = zeros(ds);         % The awkward term
    for ii = 1:model.do
        at(2*ii-1,2*ii-1) = yR(ii) * nl(2*ii-1);
        at(2*ii,2*ii) = yR(ii) * nl(2*ii);
    end
    
    if isposdef(Hess - at)
        Hess = Hess - at;
    end
    
end

end

