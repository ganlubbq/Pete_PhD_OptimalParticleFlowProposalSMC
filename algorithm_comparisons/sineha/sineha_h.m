function [ y_mn ] = sineha_h( model, x )
%ha_h Deterministic observation function for the heartbeat alignment
%model.

% Unpack state
A = x(1);
T = x(2);
tau = x(3);
omega = x(4);
phi = x(5);
B = x(6);

t = (0:model.do-1)'/model.fs;
y_mn = A * exp(-0.5*(t-tau).^2/T^2) .* sin(omega*(t-tau)+phi) + B;

end
