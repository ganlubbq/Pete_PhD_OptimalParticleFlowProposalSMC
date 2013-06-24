function [ dh_dx ] = sineha_obsjacobian( model, x )
%ha_obsjacobian Calculate observation function jacobian for the heartbeat
%alignment model.

% Unpack state
A = x(1);
T = x(2);
tau = x(3);
omega = x(4);
phi = x(5);
B = x(6);

t = (0:model.do-1)'/model.fs;

% Useful terms
exp_term = exp(-0.5*(t-tau).^2/T^2);
sin_term = exp_term .* sin(omega*(t-tau)+phi);
cos_term = exp_term .* cos(omega*(t-tau)+phi);

% Jacobian terms
dh_dA = sin_term;
dh_dT = A * sin_term .* ((t-T).^2)/(T^3);
dh_dtau = A * ( (t-tau).*sin_term/(T^2) - omega*cos_term );
dh_domega = A * (t-tau) .* cos_term;
dh_dphi = A * cos_term;
dh_dB = ones(size(t));

% Stack it
dh_dx = [dh_dA, dh_dT, dh_dtau, dh_domega, dh_dphi, dh_dB];

end

