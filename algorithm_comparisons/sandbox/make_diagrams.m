clup

%% Gaussian case

rng(0);

% Set model parameters
Q = [1 0.9; 0.9 1];
R = [0.02 -0.005; -0.005 0.01];
H = eye(2);
m = [0 0]';
y = [1 0]';

z0 = [-0.5; 0];

g1 = 0;
g2 = 0.03;
g3 = 0.3;

% Create a range of lambdas
a = 0.007;
r = 3;
lam = a; lam_rng = [0 a];
while lam < 1
    lam = r*lam;
    lam_rng = [lam_rng lam];
end
lam_rng = [lam_rng(1:end-1) 1];

% Loop through creating ellipsoids
mu_arr = zeros(2,length(lam_rng));
Sigma_arr = zeros(2,2,length(lam_rng));
for ll = 1:length(lam_rng)
    lam = lam_rng(ll);
    
    Sigma_arr(:,:,ll) = inv(inv(Q)+lam*H'*(R\H));
    mu_arr(:,ll) = Sigma_arr(:,:,ll)*(Q\m+lam*H'*(R\y));
        
end

% Create a fine grid of lambdas
lam_res = 0.01;
lam_grid = 0:lam_res:1;

% Loop through finely creating trajectories
z1 = z0;
z2 = z0;
z3 = z0;
x1_arr = zeros(2,length(lam_grid));
x2_arr = zeros(2,length(lam_grid));
x3_arr = zeros(2,length(lam_grid));
for ll = 1:length(lam_grid)
    
    lam = lam_grid(ll);
    
    Sigma = inv(inv(Q)+lam*H'*(R\H));
    mu = Sigma*(Q\m+lam*H'*(R\y));
    
    x1_arr(:,ll) = mu + sqrtm(Sigma)*z1;
    x2_arr(:,ll) = mu + sqrtm(Sigma)*z2;
    x3_arr(:,ll) = mu + sqrtm(Sigma)*z3;
    
    z1 = mvnrnd(exp(-0.5*g1*lam_res)*z1, (1-exp(-g1*lam_res)));
    z2 = mvnrnd(exp(-0.5*g2*lam_res)*z2, (1-exp(-g2*lam_res)));
    z3 = mvnrnd(exp(-0.5*g3*lam_res)*z3, (1-exp(-g3*lam_res)));
    
end

% Set up figure
figure, hold on,
lims = [-1.1 1.1];
xlim(lims); ylim(lims);

% Plot ellipsoids
for ll = 1:length(lam_rng)
    h = plot_gaussian_ellipsoid(mu_arr(:,ll), Sigma_arr(:,:,ll));
    set(h, 'color', [.5 .5 .5])%, 'linestyle', ':');
end

% Plot trajectories
plot(x1_arr(1,1), x1_arr(2,1), 'ob')
plot(x1_arr(1,:), x1_arr(2,:), ':k', 'linewidth', 1)
plot(x2_arr(1,:), x2_arr(2,:), 'b--', 'linewidth', 1)
plot(x3_arr(1,:), x3_arr(2,:), 'color', [0 .5 0], 'linewidth', 1)

% Save
matlab2tikz('filename', 'gaussian_flow.tikz',...
            'width', '4cm',...
            'height', '4cm');
xlim([0.55 0.85]);
ylim([0 0.3]);
matlab2tikz('filename', 'gaussian_flow_zoom.tikz',...
            'width', '4cm',...
            'height', '4cm');

%% Non-Gaussian Case

rng(1);

% Set up grid
x_rng = -2:0.01:2;
N = length(x_rng);
[x_coord,y_coord] = meshgrid(x_rng, x_rng);
coord = [x_coord(:), y_coord(:)]';

% Define densities
m0 = [-0.4 0.4]'; P0 = 0.5*eye(2);
y = 1; R = 0.001;

% Calculate likelihood and prior
prior = reshape((mvnpdf(coord', m0', P0)), N, N);
[b_coord, r_coord] = cart2pol(x_coord(:), y_coord(:));
r_coord = reshape(r_coord, N, N);
b_coord = reshape(b_coord, N, N);
lhood = reshape((mvnpdf(r_coord(:), y, R)), N, N);
post = prior.*lhood;

% Create a fine grid of lambdas
% lam_res = 0.001;
% lam_grid = 0:lam_res:1;
a = 1E-5;
r = 1.1;
lam = a; lam_grid = [0 a];
while lam < 1
    lam = r*lam;
    lam_grid = [lam_grid lam];
end
lam_grid = [lam_grid(1:end-1) 1];

% Initialise
x_arr = zeros(2,length(lam_grid));
% x = mvnrnd(m0, P0)';
x = [-0.5; 0];
m = m0; P = P0;

% Create figure
figure, hold on
xlim([-1.4 0.5]);
ylim([-0.5 1.4]);

% Run particle flow
g = 0.1;
for ll = 1:length(lam_grid)-1
    
    lam = lam_grid(ll);
    next_lam = lam_grid(ll+1);
    
    x_arr(:,ll) = x;
    
    % Linearise
    H = [x(1) x(2)]/sqrt(x(1)^2+x(2)^2);
    yl = y - sqrt(x(1)^2+x(2)^2) + H*x;
    
    % Gaussian approx update
    S = H*P*H'+R/(next_lam-lam);
    m_next = m + P*(H'/S)*(yl-H*m);
    P_next = P - P*(H'/S)*H*P;
    
    % State update
    Gam = exp(-0.5*g*(next_lam-lam))*sqrtm(P_next)/sqrtm(P);
    Ome = (1-exp(-g*(next_lam-lam)))*P_next;
    if g > 0
        x = m_next + Gam*(x-m) + sqrtm(Ome)*(mvnrnd([0 0], eye(2))');
    else
        x = m_next + Gam*(x-m);
    end
    
    % Move on
    m = m_next;
    P = P_next;
    
    if any(ll==[1 75 95 122])
        seq_den = prior.*lhood.^lam;
        contour(x_coord, y_coord, seq_den, 1);
        colormap(zeros(2,3));
        h = plot_gaussian_ellipsoid(m, P);
        set(h, 'color', [.5 .5 .5], 'linestyle', ':');
    end
    
end
x_arr(:,end) = x;

% Plot path
plot(x_arr(1,:), x_arr(2,:), 'color', [0 .5 0])

% Save
matlab2tikz('filename', 'approx_gaussian_flow.tikz',...
            'width', '4cm',...
            'height', '4cm');
xlim([-1.1 -0.8]);
ylim([0.05 0.35]);
matlab2tikz('filename', 'approx_gaussian_flow_zoom.tikz',...
            'width', '4cm',...
            'height', '4cm');
