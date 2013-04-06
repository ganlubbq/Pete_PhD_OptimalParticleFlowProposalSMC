function [ dH ] = ha_obsjacobian( model, x )
%ha_obsjacobian Calculate observation function jacobian for the heartbeat
%alignment model.

% Unpack state
tau = x(1);
A = x(2);

t = (0:model.do-1)'/model.fs;
n = 0:model.dw-1;

grid = (t*model.fs*ones(1,model.dw)-ones(length(t),1)*n) - tau*model.fs;
H = sinc(grid);
H(isinf(grid))=0;

dH = [-A*model.fs*dsinc(grid)*model.template, H*model.template];

end

function d = dsinc(x)

px = pi*x;
d = pi*(px.*cos(px)-sin(px))./(px.^2);

d(x==0)=0;

end