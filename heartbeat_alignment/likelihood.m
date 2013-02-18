function [ lhood ] = likelihood( model, tau, A, template, signal )
%LIKELIHOOD

t = (0:model.K-1)'/model.fs;
n = 0:model.dw-1;

grid = (t*model.fs*ones(1,model.dw)-ones(length(t),1)*n) - tau*model.fs;
H = sinc(grid);

H(isinf(grid))=0;

R = model.y_obs_vr*eye(model.K);

lhood = loggausspdf(signal', A*(H*template.m), A^2*H*template.P*H'+R);

end

