function [ H ] = sinusoidseparation_obsjacobian( model, con_state, dis_state )
% Calculate the Jacobian for the observation mean function for the
% sinusoidal separation model.

M = model.M;

tau = (0:model.do-1)'*model.Ts;

H = zeros(model.do, model.dsc);

for ii = 1:M
    
    Aii = con_state(ii);
    wii = con_state(ii+M);
    pii = con_state(ii+2*M);
    lii = dis_state(ii);
    
    H(:,ii) =      lii *              sin(wii*tau - pii);
    H(:,ii+M) =    lii * Aii * tau .* cos(wii*tau - pii);
    H(:,ii+2*M) = -lii * Aii *        cos(wii*tau - pii);
    
end


end

