function [state, ppsl_prob] = sineha_linearisedoidproposal( model, prev_state, obs, state )
%ha_linearisedoidproposal Sample and/or evaluate approximation of the OID
%for linearised around a local maximum for the heartbeat alignment model.

% If prev_state is empty, assume that this is the first step, so use the
% prior density instead of the transition density.

% Prior
if isempty(prev_state)
    prior_mn = [model.A1_mn; model.T1_mn; model.tau1_mn; model.omega1_mn; model.phi1_mn; model.B1_mn];
else
    prior_mn = prev_state;
end
prior_vr = diag([prior_mn(1)^2/model.A_shape;
                 prior_mn(2)^2/model.T_shape;
                 prior_mn(3)^2/model.tau_shape;
                 model.omega_vr;
                 model.phi_vr;
                 model.B_vr]);

% Sample a starting point
if isempty(prev_state)
    start_x = sineha_stateprior(model);
else
    start_x = sineha_transition(model,prev_state);
end

% Maximise OID
h_of = @(x) log_oid_with_derivs(model, prior_mn, obs, x);
options = optimset('GradObj','on','Display','notify-detailed');
lin_x = fminunc(h_of,start_x,options);

% % Plot track
% figure(1), hold on
% plot([start_x(1) lin_x(1)], [start_x(2) lin_x(2)]);
% plot(lin_x(1), lin_x(2), 'o');

% Laplace Approximation
[~, ~, Hess] = log_oid_with_derivs(model, prior_mn, obs, lin_x);
ppsl_mn = lin_x;
ppsl_vr = inv(Hess);
ppsl_vr = (ppsl_vr+ppsl_vr')/2;
if(~isposdef(ppsl_vr))
    ppsl_vr = prior_vr;
end

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

function [func, grad, hess] = log_oid_with_derivs(model, prior_mn, y, x)
% Negative Log-OID with gradient and Hessian

R = model.R;
h_x = sineha_h(model, x);
H = sineha_obsjacobian(model, x);
dy = y - h_x;

% Unpack state
A = x(1);
T = x(2);
tau = x(3);
omega = x(4);
phi = x(5);
B = x(6);

A_scale = prior_mn(1)/model.A_shape;
T_scale = prior_mn(2)/model.T_shape;
tau_scale = (prior_mn(3)-prior_mn(2))/model.tau_shape;

% Function itself
func_prior =  (model.A_shape-1)*log(A)-A/A_scale ...
             +(model.T_shape-1)*log(T)-T/T_scale ...
             +(model.tau_shape-1)*log(tau-T)-(tau-T)/tau_scale ...
             -(omega-prior_mn(4))^2/(2*model.omega_vr) ...
             -(phi-prior_mn(5))^2/(2*model.phi_vr) ...
             -(B-prior_mn(6))^2/(2*model.B_vr);
func_lhood = -(dy'/R)*dy/2;
func = - func_lhood - func_prior;

% Gradient
grad_prior = [(model.A_shape-1)/A-1/A_scale;
    (model.T_shape-1)/T-1/T_scale - (model.tau_shape-1)/(tau-T) + 1/tau_scale;
    (model.tau_shape-1)/(tau-T)-1/tau_scale;
    -(omega-prior_mn(4))/model.omega_vr;
    -(phi-prior_mn(5))/model.phi_vr;
    -(B-prior_mn(6))/model.B_vr];
grad_lhood = H'*(R\dy);
grad = - grad_lhood - grad_prior;

if nargout > 2
    
    hess_prior = diag([-(model.A_shape-1)/(A^2);
        -(model.T_shape-1)/(T^2) - (model.tau_shape-1)/(tau-T)^2;
        -(model.tau_shape-1)/(tau-T)^2;
        -1/model.omega_vr;
        -1/model.phi_vr;
        -1/model.B_vr]);
    off_diagional = (model.tau_shape-1)/(tau-T)^2;
    hess_prior(2,3) = off_diagional;
    hess_prior(3,2) = off_diagional;
        
    hess_lhood = - H'*(R\H);
    
    hess = - hess_lhood - hess_prior;
    
    if(~isposdef(hess))
        warning('Hessian of the OID is not positive definite at a supposed maximum');
    end
    
end

end

