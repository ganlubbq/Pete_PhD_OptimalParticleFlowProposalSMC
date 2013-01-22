function [ new_dis_state, prob ] = sinusoidseparation_discrete_transition( model, dis_state, new_dis_state )
% Sample and/or calculate transition density for the discrete part of the
%statefor sinusoidal separation model.

% prob is a log-probability.

% Sample state if not provided
if (nargin<4)||isempty(new_dis_state)
    change = binornd(1,[model.ptrans_pres*ones(model.M,1); model.ptrans_noise]);
    new_dis_state = xor(dis_state,change);
else
    new_dis_state = new_state(model.dsc+1:model.ds,1);
    change = xor(dis_state, new_dis_state);
end

% Calculate probability if required
if nargout>1
    prob = sum(log(binopdf(change,1,[model.ptrans_pres*ones(model.M,1); model.ptrans_noise])));
else
    prob = [];
end

end

