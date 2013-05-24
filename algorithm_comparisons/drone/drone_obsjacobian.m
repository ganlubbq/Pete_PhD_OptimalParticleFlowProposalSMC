function [ H ] = drone_obsjacobian( model, x )
%drone_obsjacobian Calculate observation function jacobian for the
%drone model.

% Useful intermediates
rng = sqrt(x(1)^2 + x(2)^2 + x(3)^2);
hoz_rng_sq = (x(1)^2+x(2)^2);
hoz_rng = sqrt(hoz_rng_sq);

% Terrain gradient
map = model.map;
dT_dx = [0 0]';
for hh = 1:map.num_hills
    hill_vr = map.vr(:,:,hh);
    dist = x(1:2) - map.mn(:,hh);
    mahal = dist'*(hill_vr\dist);
    if mahal < 25
        dT_dx = dT_dx - map.alt(hh)*det(2*pi*hill_vr)^(-1/2)*exp(-mahal/2)*(hill_vr\dist);
    end
end

dh1_dx = [-x(2), x(1), 0, 0, 0, 0]/hoz_rng_sq;
dh2_dx = [x(1), x(2), x(3), 0, 0, 0]/rng;
dh3_dx = [-dT_dx(1), -dT_dx(2), 1, 0, 0, 0];
dh4_dr = [x(4)*(x(2)^2+x(3)^2)-x(1)*(x(5)*x(2) + x(6)*x(3)),...
          x(5)*(x(1)^2+x(3)^2)-x(2)*(x(4)*x(1) + x(6)*x(3)),...
          x(6)*(x(1)^2+x(2)^2)-x(3)*(x(4)*x(1) + x(5)*x(2))]/(rng^3);
dh4_dv = [x(1), x(2), x(3)]/rng;

H = [dh1_dx; dh2_dx; dh3_dx; [dh4_dr dh4_dv]];
     
end

