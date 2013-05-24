function [ T ] = drone_obssecondderivtensor( model, x )
%drone_obssecondderivtensor Calculate the ds-by-ds-by-do tensor of
%second derivatives of the nonlinear observation function drone_h.

T = zeros(model.do, model.ds, model.ds);

% Uneful intermediaries
hrsq = x(1)^2 + x(2)^2;
hrng = sqrt(hrsq);
rsq = x(1)^2 + x(2)^2 + x(3)^2;
rng = sqrt(rsq);

% Bearing - depends on x(1) and x(2)
T(1,1,1) =  2*x(1)*x(2)/( hrsq^2 );
T(1,2,2) = -2*x(1)*x(2)/( hrsq^2 );
T(1,1,2) = (x(2)^2 - x(1)^2)/( hrsq^2 );
T(1,2,1) = (x(2)^2 - x(1)^2)/( hrsq^2 );

% Range - depends on x(1), x(2), and x(3)
T(2,1,1) = ( x(2)^2 + x(3)^2 )/( rng^3 );
T(2,1,2) = ( x(1)^2 + x(3)^2 )/( rng^3 );
T(2,1,3) = ( x(1)^2 + x(2)^2 )/( rng^3 );
T(2,1,2) = ( x(1)*x(2) )/( rng^3 );
T(2,1,3) = ( x(1)*x(3) )/( rng^3 );
T(2,2,3) = ( x(2)*x(3) )/( rng^3 );
T(2,2,1) = T(3,1,2);
T(2,3,1) = T(3,1,3);
T(2,3,2) = T(3,2,3);

% Altitude - depends on x(1) and x(2)

% Terrain curvature
map = model.map;
d2T_dx2 = zeros(2);
for hh = 1:map.num_hills
    hill_vr = map.vr(:,:,hh);
    dist = x(1:2) - map.mn(:,hh);
    mahal = dist'*(hill_vr\dist);
    if mahal < 25
        vr_scale_dist = hill_vr\dist;
        d2T_dx2 = d2T_dx2 + map.alt(hh)*det(2*pi*hill_vr)^(-1/2)*exp(-mahal/2)* ...
            (-hill_vr + vr_scale_dist*vr_scale_dist');
    end
end

T(3,1:2,1:2) = -d2T_dx2;

% Range Rate - depends on all terms of x... crap
% First quadrant - double postion derivatives
T(4,1,1) = ( -3*x(1)*x(4)*( x(2)^2 + x(3)^2 ) - ( x(2)*x(5) + x(3)*x(6) )*( x(2)^2 + x(3)^2 - 2*x(1)^2 ) )/( rng^5 );
T(4,2,2) = ( -3*x(2)*x(5)*( x(1)^2 + x(3)^2 ) - ( x(1)*x(4) + x(3)*x(6) )*( x(1)^2 + x(3)^2 - 2*x(2)^2 ) )/( rng^5 );
T(4,3,3) = ( -3*x(3)*x(6)*( x(1)^2 + x(2)^2 ) - ( x(1)*x(4) + x(2)*x(5) )*( x(1)^2 + x(2)^2 - 2*x(3)^2 ) )/( rng^5 );
T(4,1,2) = ( -x(2)*x(4)*( x(2)^2 + x(3)^2 - 2*x(1)^2 ) - x(1)*x(5)*( x(1)^2 + x(3)^2 - x(2)^2 ) + 3*x(1)*x(2)*x(3)*x(6) )/( rng^5 );
T(4,1,3) = ( -x(3)*x(4)*( x(2)^2 + x(3)^2 - 2*x(1)^2 ) - x(1)*x(6)*( x(1)^2 + x(2)^2 - x(3)^2 ) + 3*x(1)*x(2)*x(3)*x(5) )/( rng^5 );
T(4,2,3) = ( -x(3)*x(5)*( x(1)^2 + x(3)^2 - 2*x(2)^2 ) - x(2)*x(6)*( x(1)^2 + x(2)^2 - x(3)^2 ) + 3*x(1)*x(2)*x(3)*x(4) )/( rng^5 );
T(4,2,1) = T(4,1,2);
T(4,3,1) = T(4,1,3);
T(4,3,2) = T(4,2,3);

% Second and third quadrants - cross postion/velocity derivatives
T(4,1,4) = ( x(2)^2 + x(3)^2 )/( rng^3 );
T(4,2,5) = ( x(1)^2 + x(3)^2 )/( rng^3 );
T(4,3,6) = ( x(1)^2 + x(2)^2 )/( rng^3 );
T(4,1,5) = ( x(1)*x(2) )/( rng^3 );
T(4,1,6) = ( x(1)*x(3) )/( rng^3 );
T(4,2,6) = ( x(2)*x(3) )/( rng^3 );
T(4,2,4) = T(4,1,5);
T(4,3,4) = T(4,1,6);
T(4,3,5) = T(4,2,6);
T(4,4:6,1:3) = squeeze(T(4,1:3,4:6))';

% Last quadrant - double velocity derivatives - are all 0, mercifully


end

