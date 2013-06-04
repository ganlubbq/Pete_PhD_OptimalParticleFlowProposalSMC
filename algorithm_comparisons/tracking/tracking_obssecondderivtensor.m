function [ T ] = tracking_obssecondderivtensor( model, x )
%tracking_obssecondderivtensor Calculate the ds-by-ds-by-do tensor of
%second derivatives of the nonlinear observation function tracking_h.

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

% Elevation - depends on x(1), x(2), and x(3)
T(2,1,1) = x(3)*( 2*x(1)^4 + x(1)^2*x(2)^2 - x(2)^4 - x(2)^2*x(3)^2 )/( hrng^3 * rsq^2 );
T(2,2,2) = x(3)*( 2*x(2)^4 + x(1)^2*x(2)^2 - x(1)^4 - x(1)^2*x(3)^2 )/( hrng^3 * rsq^2 );
T(2,3,3) = -2*x(3)*hrng/( rsq^2 );
T(2,1,2) = x(1)*x(2)*x(3)*(3*x(1)^2 + 3*x(2)^2 + x(3)^2)/( hrng^3 * rsq^2 );
T(2,1,3) = x(1)*(x(3)^2 - x(1)^2 - x(2)^2)/( hrng * rsq^2 );
T(2,2,3) = x(2)*(x(3)^2 - x(1)^2 - x(2)^2)/( hrng * rsq^2 );
T(2,2,1) = T(2,1,2);
T(2,3,1) = T(2,1,3);
T(2,3,2) = T(2,2,3);

% Range - depends on x(1), x(2), and x(3)
T(3,1,1) = ( x(2)^2 + x(3)^2 )/( rng^3 );
T(3,1,2) = ( x(1)^2 + x(3)^2 )/( rng^3 );
T(3,1,3) = ( x(1)^2 + x(2)^2 )/( rng^3 );
T(3,1,2) = ( x(1)*x(2) )/( rng^3 );
T(3,1,3) = ( x(1)*x(3) )/( rng^3 );
T(3,2,3) = ( x(2)*x(3) )/( rng^3 );
T(3,2,1) = T(3,1,2);
T(3,3,1) = T(3,1,3);
T(3,3,2) = T(3,2,3);

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

