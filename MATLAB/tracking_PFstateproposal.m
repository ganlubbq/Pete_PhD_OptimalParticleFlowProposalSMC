function [ state, prob ] = tracking_PFstateproposal( algo, model, prev_kk, prev_state, observ )
%NLBENCHMARK_STATEPROPOSAL Sample and/or calculate proposal density for
%2D tracking. This uses the particle flow approximation to the OID.

% State prior
prior_mn = model.A*prev_state;

% Sample it
[state, prob] = tracking_transition(model, prev_kk, prev_state);

% if prev_kk > 2
%     th = 0:pi/720:2*pi;
%     [X, Y] = pol2cart(th, (observ(2)+2*sqrt(model.R(2,2)))*ones(size(th)));
%     figure(1), hold on
%     plot(X, Y, '--r')
% end

% Flow integration loop
for ll = 1:algo.L
    
    % Step size
    dl = 1/algo.L;
    lam = (ll-1)*dl;
%     lam = (1-2^-(ll-1));
%     dl = (1-2^-(ll)) - (1-2^-(ll-1));
%     if ll == algo.L
%         dl = 2^-(ll-1);
%     end
    
    % Predict
    v = calc_particle_velocity(model, lam, state, observ, prior_mn);
    pred_state = state + v*dl;
    
%     % Update
    for ii = 1:1
        pred_v = calc_particle_velocity(model, lam+dl, pred_state, observ, prior_mn);
        pred_state = state + 0.5*dl*(v+pred_v);
    end
    
    state = pred_state;
    
%     if prev_kk > 2
%         figure(1), hold on, plot(state(1), state(2), 'xb')
%     end
    
end

end

function v = calc_particle_velocity(model, lam, x, y, m)

% Shorter variable names
Q = model.Q;
R = model.R;
ds = model.ds;

% Linearise
H = tracking_hessian(x);

% % Find particle velocity using Gaussian approximation
% y_mod = y - tracking_h(model, x) + H*x;
% A = -0.5*Q*H'*((R+lam*H*Q*H')\H);
% b = (eye(ds)+2*lam*A)*((eye(ds)+lam*A)*Q*H'*(R\y_mod)+A*m);
% v = A*x+b;
    
%     % Find particle velocity using Incompresible flow
%     y_mn = tracking_h(model, x);
%     dlogp_dx = lam*H'*(R\(y-y_mn)) - Q\(x-m);
%     loglhood = loggausspdf(y, y_mn, R);
%     norm_const = (dlogp_dx'*dlogp_dx);
%     corr_mat = eye(ds) - dlogp_dx*dlogp_dx'/norm_const;
% %     if norm_const > 0.001;
%         v = -loglhood*dlogp_dx/norm_const - corr_mat*x;
% %     else
% %         v = zeros(model.ds, 1);
% %     end

    % Find particle velocity using small curvature flow
    A = -0.5*Q*H'*((R+lam*H*Q*H')\H);
    y_mn = tracking_h(model, x);
    dlogp_dx = lam*H'*(R\(y-y_mn)) - Q\(x-m);
    ddlogp_ddx = -(lam*H'*(R\H) + inv(Q));
    dloglhood_dx = H'*(R\(y-y_mn));
    v = -ddlogp_ddx \ ( dloglhood_dx + A'*dlogp_dx );


end
