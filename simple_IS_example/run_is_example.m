clup
dbstop if error
rng(1);

% Define model
model.ds = 2; ds = model.ds;
model.m0 = ones(ds,1);
model.P0 = eye(ds);
model.y = sqrt(ds);
model.R = 0.1^2;
model.h = @(x) sqrt(sum(x.^2, 1));
model.H = @(x) x'/model.h(x);

% Probability evaluations
eval_prior = @(x) loggausspdf(x, model.m0, model.P0);
eval_lhood = @(x) loggausspdf(model.y, model.h(x), model.R);

% Algorithm parameters
algo.Np = 30; Np = algo.Np;
algo.g = 0.03;

% Pseudo-time grid parameters
a = 1E-5; r = 1.1;

% Make grid
lam = a; lam_grid = [a];
while lam < 1
    lam = r*lam;
    lam_grid = [lam_grid lam];
end
lam_grid = [lam_grid(1:end-1) 1];
% lam_grid = [1E-5:1E-5:1E-4 2E-4:1E-4:1E-3 2E-3:1E-3:1E-2 2E-2:1E-2:1];

% Create particle array
x_arr = zeros(ds,Np);
w_arr = zeros(1,Np);

% Create figures
figure(1), hold on, xlim([-2 3]), ylim([-2 3])
figure(2), hold on

% Add contours
x_rng = -2:0.01:2;
[x_coord,y_coord] = meshgrid(x_rng, x_rng);
coord = [x_coord(:), y_coord(:)]';
prior_grid = reshape(log(mvnpdf(coord', model.m0', model.P0)), length(x_rng), length(x_rng));
lhood_grid = reshape(log(mvnpdf(model.y, model.h(coord)', model.R)), length(x_rng), length(x_rng));
post_grid = exp(prior_grid + lhood_grid);
figure(1), h=contour(x_coord, y_coord, post_grid, 3); 
colormap(zeros(2,3));

% Loop through particles
for ii = 1:Np
    
    fprintf(1, 'Particle number %u.\n', ii);
    
    % Initialise particle stuff
    x = mvnrnd(model.m0', model.P0)';
    w = 0;
    lam = 0;
    m = model.m0;
    P = model.P0;
    prior = eval_prior(x);
    lhood = eval_lhood(x);
    postapprox = prior;
    x_traj = zeros(ds, length(lam_grid)+1);
    x_traj(:,1) = x;
    w_traj = zeros(1, length(lam_grid)+1);
    w_traj(1) = w;
    
    % Plot initial state
    figure(1), plot(x(1), x(2), 'ob');
    
    % Loop through pseudo-time grid
    for ll = 1:length(lam_grid);
        
        % Update pseudo-time
        dl = lam_grid(ll)-lam;
        lam = lam_grid(ll);
        
        % Linearise
        H = model.H(x);
        yl = model.y - model.h(x) + H*x;
        
        % Gaussian approx update
        S = H*P*H'+model.R/dl;
        m_next = m + P*(H'/S)*(yl-H*m);
        P_next = P - P*(H'/S)*H*P;
%         P_next = inv(inv(model.P0) + lam*H'*(model.R\H));
%         m_next = P_next*(model.P0\model.m0 + lam*H'*(model.R\yl));
        
        % State and weight update
        Gam = exp(-0.5*algo.g*dl)*sqrtm(P_next)/sqrtm(P);
        Ome = (1-exp(-algo.g*dl))*P_next;
        Ome = (Ome+Ome')/2;
        if algo.g > 0
            x =  mvnrnd( (m_next + Gam*(x-m))', Ome )';
            next_prior = eval_prior(x);
            next_lhood = eval_lhood(x);
            next_postapprox = loggausspdf(x, m, P);
            w = w + (next_prior-prior) + (lam*next_lhood-(lam-dl)*lhood)...
                  + (postapprox-next_postapprox);
        else
            x = m_next + Gam*(x-m);
            next_prior = eval_prior(x);
            next_lhood = eval_lhood(x);
            w = w + (next_prior-prior) + (lam*next_lhood-(lam-dl)*lhood) + log(det(Gam));
        end
        
        % Store state
        x_traj(:,ll+1) = x;
        w_traj(:,ll+1) = w;
        
        % Move on
        m = m_next;
        P = P_next;
        lhood = next_lhood;
        prior = next_prior;
        postapprox = next_postapprox;
        
    end
    
    % Save final values
    x_arr(:,ii) = x;
    w_arr(ii) = w;
    
    % Plot particle
    figure(1), plot(x_traj(1,:), x_traj(2,:), ':k');
    figure(2), plot(w_traj(1,:), 'k');
    
    figure(1), plot(x(1), x(2), 'xr');
    
end

ess = calc_ESS(w_arr)

%% Compare with Gaussian importance density
H = model.H([1; 1]);
Sigma = inv(inv(model.P0)+H'*(model.R\H));
mu = Sigma*(model.P0\model.m0+H'*(model.R\model.y));
xis_arr = zeros(ds,Np);
wis_arr = zeros(1,Np);
for ii = 1:Np
    x = mvnrnd(mu', Sigma)';
    xis_arr(:,ii) = x;
    wis_arr(ii) = eval_prior(x) + eval_lhood(x) - loggausspdf(x, mu, Sigma);
end

is_ess = calc_ESS(wis_arr)

%% Save
matlab2tikz('filename', 'gaussian_flow_importance_sampling.tikz',...
            'width', '4cm',...
            'height', '4cm');