function [ state, prob ] = sineha_stateprior( model, state )
%ha_stateprior Sample and/or evaluate observation density for the heartbeat
%alignment model.

% prob is a log-probability.

% Sample state if not provided
if (nargin<2)||isempty(state)
    A = gamrnd(model.A_shape, model.A1_mn/model.A_shape);
    T = gamrnd(model.T_shape, model.T1_mn/model.T_shape);
    tau = T + gamrnd(model.tau_shape, (model.tau1_mn-model.T1_mn)/model.tau_shape);
    omega = mvnrnd(model.omega1_mn, model.omega_vr);
    phi = mvnrnd(model.phi1_mn, model.phi_vr);
    B = mvnrnd(model.B1_mn, model.B_vr);

    state = [A; T; tau; omega; phi; B];
else
    A = state(1);
    T = state(2);
    tau = state(3);
    omega = state(4);
    phi = state(5);
    B = state(6);
end

% Calculate probability if required
if nargout>1
    if tau>T
        prob = 0;
        prob = prob + log(gampdf(A, model.A_shape, model.A1_mn/model.A_shape));
        prob = prob + log(gampdf(T, model.T_shape, model.T1_mn/model.T_shape));
        prob = prob + log(gampdf(tau-T, model.tau_shape, (model.tau1_mn-model.T1_mn)/model.tau_shape));
        prob = prob + loggausspdf(omega, model.omega1_mn, model.omega_vr);
        prob = prob + loggausspdf(phi, model.phi1_mn, model.phi_vr);
        prob = prob + loggausspdf(B, model.B1_mn, model.B_vr);
    else
        prob = -inf;
    end
else
    prob = [];
end

end

