% This makes pretty pictures of particle evolution during a smooth update
% using the linear Gaussian model. Stick a breakpoint somewhere in
% lg_smoothupdate and run this.

% IT WILL MANGLE OPERATION, so don't try to carry on (unless you feel like
% encapsulating this in a function)

% Number of pseudo-time divisions
L = 100;

% Create arrays
state1 = zeros(algo.N, L+1);
state2 = zeros(algo.N, L+1);
weight = zeros(algo.N, L+1);

% Particle loop
for ii = 1:algo.N
    
    state1(ii,1) = 
    
end