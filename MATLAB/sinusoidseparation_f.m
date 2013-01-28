function [ next_state_mn ] = sinusoidseparation_f( model, kk, state )
% Nonlinear function giving the predictive mean of the next
%state for the sinusoid separation model. Continuous part only

M = model.M;

A = state(1:M);
w = state(M+1:2*M);
p = state(2*M+1:3*M);

A_mn = model.beta1 * A + model.beta2 * (sum(A)/(1+sum(A)^2));
w_mn = model.beta3 * w + model.beta4 * prod(w)^(1/M);
% w_mn = max(w_mn, model.wmin);
p_mn = model.beta5 * abs(p).^model.beta6;

next_state_mn = [A_mn; w_mn; p_mn];

end

