function [ obs_mn ] = sinusoidseparation_h( model, con_state, dis_state )
% Nonlinear function giving the mean of the current observation for the
%sinusoid separation model.

M = model.M;

obs_mn = zeros(model.do,1);

tau = (0:model.do-1)'*model.Ts;

for ii = 1:model.M
    
    obs_mn = obs_mn + dis_state(ii) * con_state(ii).*sin(con_state(ii+M)*tau-con_state(ii+2*M));
    
end

end
