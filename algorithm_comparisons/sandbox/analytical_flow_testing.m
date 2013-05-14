clup

ds = 2;
I = eye(ds);
P = eye(ds);
m = -5*ones(ds,1);

do = 1;
H = [1.5 0.5];
y = 20;

% do = 3;
% H = [1 0; 0 1; 0.5 0.5];
% y = [10; 10; 10.3];

R = 0.1*eye(do)+0.05*ones(do);

Dscale = 1;

dl = 0.01;1;
lam_rng = 0:dl:1;
N = 100;

Sigma = inv(inv(P)+(H'/R)*H);
mu = Sigma*(P\m + (H'/R)*y);
figure(1), hold on
h1 = plot_gaussian_ellipsoid(m, P, 2);
h2 = plot_gaussian_ellipsoid(mu, Sigma, 2); set(h2,'color','r');

figure(2), hold on
figure(3), hold on

for ii = 1:N
    
    % Sample prior
    x0 = mvnrnd(m', P)';
    
    x_rng = zeros(ds, length(lam_rng));
    x_rng(:,1) = x0;
    
    x = x0;
    
    figure(1), plot(x0(1), x0(2), 'bx')
    
    % Loop through time
    for ll = 1:length(lam_rng)-1
        
        lam0 = lam_rng(ll);
        lam = lam_rng(ll+1);
        
        [ x, S] = linear_flow_move( lam, lam0, x, m, P, y, H, R, Dscale );
        
        x_rng(:,ll+1) = x;
        
    end
    
    figure(1), plot(x(1), x(2), 'rx')
    figure(2), plot(x_rng(1,:), x_rng(2,:), 'b');
    figure(3), plot(lam_rng, x_rng(1,:))
    
end

figure(2)
plot_gaussian_ellipsoid(m, P, 2);
plot_gaussian_ellipsoid(mu, Sigma, 2);

