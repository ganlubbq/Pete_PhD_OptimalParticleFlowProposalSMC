function [ x, ppsl_prob ] = lg_move( x0, m, P, G, Phi, nu, Omega )
%lg_move Calculate the movement produced by a linear flow for a
%linear Guassain model, given various pre-calculated values (which are
%common to all particles).

% New x mean
x_mn = G*x0 + Phi*m + nu;

% Stochastic bit
if ~isempty(Omega)
    x = mvnrnd(x_mn', Omega)';
else
    x = x_mn;
end

if ~isempty(Omega)
    % Proposal probability
    ppsl_prob = loggausspdf(x, (G+Phi)*m+nu, G*P*G'+Omega);
else
    ppsl_prob = 0;
end

end
