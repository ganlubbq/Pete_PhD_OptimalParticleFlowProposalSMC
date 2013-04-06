function [ y_mn ] = ha_h( model, x )
%ha_h Deterministic observation function for the heartbeat alignment
%model.

% Unpack state
tau = x(1);
A = x(2);

t = (0:model.do-1)'/model.fs;
n = 0:model.dw-1;
grid = (t*model.fs*ones(1,model.dw)-ones(length(t),1)*n) - tau*model.fs;
H = sinc(grid);
H(isinf(grid))=0;

y_mn = A*H*model.template;

end
