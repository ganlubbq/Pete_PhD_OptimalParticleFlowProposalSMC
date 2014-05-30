clup
dbstop if error

flag_plot = true;
file_path = 'CAMSAP_plots/';

%% Simple pathological example

% Set up grid
x_rng = -2:0.01:2;
N = length(x_rng);
[x_coord,y_coord] = meshgrid(x_rng, x_rng);
coord = [x_coord(:), y_coord(:)]';

% Define densities
m = [-0.4 0.4]'; P = 0.5*eye(2);
y = 1; R = 0.001;

% Calculate likelihood and prior
prior = reshape((mvnpdf(coord', m', P)), N, N);
[b_coord, r_coord] = cart2pol(x_coord(:), y_coord(:));
r_coord = reshape(r_coord, N, N);
b_coord = reshape(b_coord, N, N);
lhood = reshape((mvnpdf(r_coord(:), y, R)), N, N);
post = prior.*lhood;

% Plot 'em
if flag_plot
    figure, hold on
%     mesh(x_coord, y_coord, prior); shading interp;
    imagesc(x_rng, x_rng, prior); xlim([-2 2]); ylim([-2 2]);
    export_pdf(gcf, [file_path 'path_ex_prior.pdf'], 8, 6);
    figure, hold on
%     mesh(x_coord, y_coord, lhood); shading interp;
    imagesc(x_rng, x_rng, lhood); xlim([-2 2]); ylim([-2 2]);
    export_pdf(gcf, [file_path 'path_ex_lhood.pdf'], 8, 6);
    figure, hold on
%     mesh(x_coord, y_coord, prior.*lhood); shading interp;
    imagesc(x_rng, x_rng, prior.*lhood); xlim([-2 2]); ylim([-2 2]);
    export_pdf(gcf, [file_path 'path_ex_post.pdf'], 8, 6);
    figure, hold on
    contour(x_coord, y_coord, post);
    export_pdf(gcf, [file_path 'path_ex_post_contour.pdf'], 8, 6);
end

%% Algorithms running on pathological example

% EKF

% Jacobian
rng_sq = m(1)^2 + m(2)^2;
rng = sqrt(rng_sq);
H = [m(1), m(2)]/rng;

% OID moments
Sigma = inv(inv(P) + H'*(R\H));
mu = Sigma*(P\m + H'*(R\y));

% Calculate density
EKF_approx = reshape((mvnpdf(coord', mu', Sigma)), N, N);

% Plot it
if flag_plot
    figure, hold on
    contour(x_coord, y_coord, EKF_approx);
    export_pdf(gcf, [file_path, 'path_ex_ekf.pdf'], 8, 6);
end

% UKF

% OID moments
h = @(x,y) sqrt(x(1)^2+x(2)^2);
[mu, Sigma] = ukf_update1(m, P, y, h, R);

% Calculate density
UKF_approx = reshape((mvnpdf(coord', mu', Sigma)), N, N);

% Plot it
if flag_plot
    figure, hold on
    contour(x_coord, y_coord, UKF_approx);
    export_pdf(gcf, [file_path, 'path_ex_ukf.pdf'], 8, 6);
end

% Gaussian Local Maximum

% Maximise
x_max = [-0.7060, 0.7060]';

% Jacobian
rng_sq = x_max(1)^2 + x_max(2)^2;
rng = sqrt(rng_sq);
H = [x_max(1), x_max(2)]/rng;

% Hessian
H2 = [x_max(2)^2, -x_max(1)*x_max(2);
      -x_max(1)*x_max(2), x_max(1)^2];

% OID moments
Sigma = inv(inv(P) + H'*(R\H) - H2*(y-sqrt(x_max(1)^2+x_max(2)^2)));
mu = x_max;

% Calculate density
GLM_approx = reshape((mvnpdf(coord', mu', Sigma)), N, N);

% Plot it
if flag_plot
    figure, hold on
    contour(x_coord, y_coord, GLM_approx);
    export_pdf(gcf, [file_path, 'path_ex_glm.pdf'], 8, 6);
end

%% PLG OID evolution

Q = [1 0.9; 0.9 1];
R = [0.02 -0.005; -0.005 0.01];
H = eye(2);
m = [0 0]';
y = [1 2]';

z = [-0.5; 0];

rot = 5;
Ups = [0 rot; -rot 0];

if flag_plot
    figure, hold on,
    lims = [-1.5 2.5];
    xlim(lims); ylim(lims);
end

a = 0.007;
r = 3;
lam = a; lam_rng = [0 a];
while lam < 1
    lam = r*lam;
    lam_rng = [lam_rng lam];
end
lam_rng = [lam_rng(1:end-1) 1];

x_rng = zeros(2,0);
for ll = 1:length(lam_rng)
    lam = lam_rng(ll);
    
    Sigma=inv(inv(Q)+lam*H'*(R\H));
    mu = Sigma*(Q\m+lam*H'*(R\y));
    x = mu + sqrtm(Sigma)*z;
    xr = mu + sqrtm(Sigma)*expm(Ups*lam)*z;
    
    if flag_plot
        plot_gaussian_ellipsoid(mu, Sigma);
    end
%     plot(mu(1), mu(2), 'x');
%     plot(x(1), x(2), 'go');
%     plot(xr(1), xr(2), 'ro');
    x_rng = [x_rng, x];
    
    d=x-mu;
    acos(d(1)/sqrt(sum(d.^2)));
    
end

if flag_plot
    plot(x_rng(1,:), x_rng(2,:), 'rx:', 'markersize', 10)
    export_pdf(gcf, [file_path, 'plg_oid_evolution.pdf'], 8, 6);
end

%%