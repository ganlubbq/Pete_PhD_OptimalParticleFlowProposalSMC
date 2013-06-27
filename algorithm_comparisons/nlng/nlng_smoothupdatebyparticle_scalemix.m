function [ state, ppsl_prob ] = nlng_smoothupdatebyparticle_scalemix( display, algo, model, fh, prev_state, obs )
%nlng_smoothupdatebyparticle Apply a smooth update for the nonlinear non-Gaussian
%benchmark model for a single particle (which means no intermediate
%resampling, but step size control is easier.)

Dscale = algo.Dscale;

dl_start = 1E-3;
dl_min = 1E-7;
dl_max = 0.5;
err_thresh = 0.01;
dl_sf = 0.8;
dl_pow = 0.7;

% Prior
if isempty(prev_state)
    m = model.m1;
    P = model.P1;
else
    prev_kk = prev_state(1);
    prev_x = prev_state(2:end);
    m = nlng_f(model, prev_kk, prev_x);
    P = model.Q;
end

% Sample initial state
if ~isempty(prev_state)
    [init_state, init_trans_prob] = feval(fh.transition, model, prev_state);
else
    [init_state, init_trans_prob] = feval(fh.stateprior, model);
end

% SMoN scaling.
if ~isinf(model.dfy)
    mix = chi2rnd(model.dfy);
else
    mix = 1;
end

% Initialise evolution arrays
dl_evo = dl_start;
lam_evo = 0;
err_evo = 0;
state_evo = init_state;
mix_evo = mix;
post_prob_evo = init_trans_prob;
ppsl_prob_evo = init_trans_prob;

% Initialise loop variables
state = init_state;
post_prob = init_trans_prob;
ppsl_prob = init_trans_prob;
dl = dl_start;
lam = 0;

% Sample perturbation
if algo.Dscale > 0
    zD = mvnrnd(zeros(ds,1)',eye(model.ds-1))';
else
    zD = zeros(model.ds-1,1);
end

% Loop
while lam < 1
    
    % Pseudo-time and step-size
    lam0 = lam;
    lam1 = lam + dl;
    if lam1 > 1
        lam1 = 1;
    end
    
    % Starting point
    x0 = state(2:end);
    kk = state(1);
    xi0 = mix;
    
    % Observation mean
    obs_mn = nlng_h(model, x0);
    
    % Linearise observation model around the current point
    H = nlng_obsjacobian(model, x0);
    y = obs - obs_mn + H*x0;
    R = model.R/xi0;
    
    % Analytical flow
    [ x, prob_ratio, drift, diffuse] = linear_flow_move( lam1, lam0, x0, m, P, y, H, R, algo.Dscale, zD );
    
    % Error estimate
    H_new = nlng_obsjacobian(model, x);
    y_new = obs - nlng_h(model, x) + H_new*x;
    drift_new = linear_drift( lam1, x, m, P, y_new, H_new, R, algo.Dscale );
    err_est = 0.5*dl*( drift_new - drift );
    
    % Step size adjustment
    err_crit = norm(err_est, 2);
    dl = min(dl_max, min(10*dl, dl_sf * (err_thresh/err_crit)^dl_pow * dl));
    if dl < dl_min
%         dl = dl_min;
%         warning('nlng_smoothupdatebyparticle:ErrorTolerance', 'Minimum step size reached. Local error tolerance exceeded.');
        break;
    end
    
    % Accept/reject step
    if err_crit < err_thresh
        
        if ~isinf(model.dfy)
%             % Mixing parameter update
%             dyR0 = (obs-nlng_h(model, x0))'*(model.R\(obs-nlng_h(model, x0)));
%             dyR1 = (obs-nlng_h(model, x))'*(model.R\(obs-nlng_h(model, x)));
%             gama1 = (model.dfy+lam1*model.do)/2;
%             gama0 = (model.dfy+lam0*model.do)/2;
%             gamb1 = 2/(1+lam1*dyR1);
%             gamb0 = 2/(1+lam0*dyR0);
            
            %         xi = gamrnd(gama1,gamb1);
            %         xi_prob_ratio = gampdf(xi,gama1,gamb1)/gampdf(xi0,gama0,gamb0);
            
%             xi = chi2rnd(model.dfy);
%             xi_prob_ratio = 1;
            
            %         xi_upd = gamrnd((gama1-gama0),gamb1);
            %         xi = (gamb1/gamb0)*xi0 + xi_upd;
            %         xi_prob_ratio = gampdf(xi,gama1,gamb1)/gampdf(xi0,gama0,gamb0);
            
%             xi_post = log(chi2pdf(xi, model.dfy));

%             xi_mhppsl = chi2rnd(model.dfy);
%             old_prob = (0.5*(model.dfy+(lam1-1)*model.do)-1)*log(xi0)       - xi0/2       + loggausspdf(obs, nlng_h(model, m), model.R/(lam1*xi0)+H_new*P*H_new');
%             new_prob = (0.5*(model.dfy+(lam1-1)*model.do)-1)*log(xi_mhppsl) - xi_mhppsl/2 + loggausspdf(obs, nlng_h(model, m), model.R/(lam1*xi_mhppsl)+H_new*P*H_new');
%             if log(rand)<((new_prob-old_prob)-(new_ppsl-old_ppsl))
%                 xi = xi_mhppsl;
%             else
%                 xi = xi0;
%             end

            for mm = 1:1
                dec = 0.1; dfy = model.dfy;
                xi_mhppsl = (1-dec)*xi0+gamrnd(0.5*dec*dfy/(1-dec),2*(1-dec));
                new_prob = log(chi2pdf(xi_mhppsl, model.dfy));
                old_prob = log(chi2pdf(xi0, model.dfy));
                new_ppsl = log(gampdf(xi_mhppsl-(1-dec)*xi0, 0.5*dec*dfy/(1-dec), 2*(1-dec)));
                old_ppsl = log(gampdf(xi0-(1-dec)*xi_mhppsl, 0.5*dec*dfy/(1-dec), 2*(1-dec)));
                if log(rand)<((new_prob-old_prob)-(new_ppsl-old_ppsl))
                    xi = xi_mhppsl;
                else
                    xi = xi0;
                end
            end
            xi_post = 0;
            xi_prob_ratio = 1;
            
        else
            xi = 1;
            xi_post = 0;
            xi_prob_ratio = 1;
        end
        
        % Update time
        lam = lam1;
        
        % Update state
        state = [kk; x];
        mix = xi;
        
        % Sample perturbation
        if algo.Dscale > 0
            zD = mvnrnd(zeros(ds,1)',eye(model.ds-1))';
        else
            zD = zeros(model.ds-1,1);
        end
        
        % Densities
        if ~isempty(prev_state)
            [~, trans_prob] = feval(fh.transition, model, prev_state, state);
        else
            [~, trans_prob] = feval(fh.stateprior, model, state);
        end
        [~, lhood_prob] = feval(fh.observation, model, state, obs);
        
        % Update probabilities
        post_prob = trans_prob + lam*lhood_prob + xi_post;
        ppsl_prob = ppsl_prob - log(prob_ratio) - log(xi_prob_ratio);
        
        % Update evolution
        dl_evo = [dl_evo dl];
        lam_evo = [lam_evo lam];
        err_evo = [err_evo err_crit];
        state_evo = [state_evo state];
        mix_evo = [mix_evo mix];
        post_prob_evo = [post_prob_evo post_prob];
        ppsl_prob_evo = [ppsl_prob_evo ppsl_prob];
        
    else
        
%         disp('Error too large. Reducing step size');
        
    end
    
end

% Plotting
if display.plot_particle_paths
    
    % 2D state trajectories
    figure(display.h_ppp(1));
    plot(state_evo(2,:), state_evo(3,:));
    plot(init_state(2), init_state(3), 'o');
    plot(state(2), state(3), 'xr');
    
    % 1D state trajectories
    figure(display.h_ppp(2));
    plot(lam_evo, state_evo(3,:));
    
    % Step size
    figure(display.h_ppp(3));
    plot(lam_evo, dl_evo);
    
    % Error estimates
    figure(display.h_ppp(4));
    plot(lam_evo, err_evo);
    
    % Probability estimate
    figure(display.h_ppp(5));
    plot(lam_evo, post_prob_evo-ppsl_prob_evo);
    %
    % Mixing variable
    figure(display.h_ppp(6));
    plot(lam_evo, mix_evo);
    
end

end

