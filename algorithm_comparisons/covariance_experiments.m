Q = [1 0.9; 0.9 1];
R = [0.02 -0.005; -0.005 0.01];
H = eye(2);
m = [0 0]';
y = [1 2]';

z = [-0.5; 0];

rot = 5;
Ups = [0 rot; -rot 0];

figure, hold on,
lims = [-1.5 2.5];
xlim(lims); ylim(lims);

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
    
    plot_gaussian_ellipsoid(mu, Sigma);
%     plot(mu(1), mu(2), 'x');
%     plot(x(1), x(2), 'go');
%     plot(xr(1), xr(2), 'ro');
    x_rng = [x_rng, x];
    
    d=x-mu;
    acos(d(1)/sqrt(sum(d.^2)));
    
end

plot(x_rng(1,:), x_rng(2,:), 'rx:', 'markersize', 10)