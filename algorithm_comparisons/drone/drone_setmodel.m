function model = drone_setmodel(test)

% Model parameters

% Using a 3D near constant velocity model and observations of range and
% bearing to a beacon, height above mapped terrain.

%%%%%%%%%%%%%%%%

% General things
model.K = 20;                   % Number of time points
model.ds = 6;                   % Dimension of the state
model.do = 4;                   % Dimension of the observations

% Parameters
sigx     = 100^2;
sigtheta = ( 10*(pi/180) )^2;   % Bearing variance
sigalt   = 0.1^2;               % Altitude variance
sigr     = 0.1^2;               % Range variance
sigrr    = 1^2;                 % Range rate variance

% Matrixes
T = 1;                          % Sampling period
a = 0.3;                        % Ensures stability
model.A = [exp(-a*T) 0 0 T 0 0; 0 exp(-a*T) 0 0 T 0; 0 0 1 0 0 T;
           0 0 0 exp(-a*T) 0 0; 0 0 0 0 exp(-a*T) 0; 0 0 0 0 0 exp(-a*T)];
model.Q = sigx * ...
    [T^3/3  0      0      T^2/2  0      0    ;
     0      T^3/3  0      0      T^2/2  0    ;
     0      0      T^3/3  0      0      T^2/2;
     T^2/2  0      0      T      0      0    ;
     0      T^2/2  0      0      T      0    ;
     0      0      T^2/2  0      0      T    ];
model.R = diag([sigtheta sigr sigalt sigrr]);

% Transition tail heaviness
model.dfx = test.STdof;

% x1 distribution
model.m1 = [50 50 50 100 0 0]';
model.P1 = diag([100 100 100 10 10 10]);

% Terrain map
num_hills = 20;
model.map.num_hills = num_hills;
model.map.alt = zeros(1,num_hills);
model.map.mn = zeros(2,num_hills);
model.map.vr = zeros(2,2,num_hills);
for hh = 1:num_hills
    model.map.alt(hh) = 10000*chi2rnd(3);
    model.map.mn(:,hh) = mvnrnd(zeros(2,1), 10000*eye(2));
    model.map.vr(:,:,hh) = wishrnd(100*eye(2), 5);
end

end