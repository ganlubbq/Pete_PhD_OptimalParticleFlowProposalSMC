function [ y_mn ] = drone_h( model, x )
%drone_h Deterministic observation function for the drone model

y_mn = zeros(model.do,1);

% Bearing
y_mn(1) = atan2(x(2), x(1));

% Range
range = sqrt( x(1)^2 + x(2)^2 + x(3)^2 );
y_mn(2) = range;

% Altitude
height = drone_terrainheight(model, x);
y_mn(3) = x(3) - height;

% Range Rate
r = x(1:3);
v = x(4:6);
y_mn(4) = dot(r,v)/range;

end
