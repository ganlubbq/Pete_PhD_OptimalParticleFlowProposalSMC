function [ T ] = nlng_obssecondderivtensor( model, x )
%nlng_obssecondderivtensor Calculate the ds-by-ds-by-do tensor of
%second derivatives of the nonlinear observation function nlng_h.

T = zeros(model.ds-1, model.ds-1, model.do);

for ii = 1:model.do
    
    T(2*ii-1, 2*ii-1, ii) = 1;
    T(2*ii, 2*ii, ii) = 1;
    
end

T = T*2*model.alpha1;


end

