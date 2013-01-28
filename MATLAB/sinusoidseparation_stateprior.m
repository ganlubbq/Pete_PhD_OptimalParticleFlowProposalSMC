function [ state, prob ] = sinusoidseparation_stateprior( model, state )
% Sample and/or calculate state prior density for sinusoid separation
% model.

% prob is a log-probability.

M = model.M;

% Sample state if not provided
if (nargin<2)||isempty(state)
    con_state = zeros(size(model.x1_lin_mn));
    while any(con_state(1:M)<model.Amin)||any(con_state(M+1:2*M)<model.wmin)
        con_state = mvnrnd(model.x1_lin_mn', model.x1_lin_vr)';
    end
    dis_state = binornd(1,[model.pprior_pres*ones(model.M,1); model.pprior_noise]);
    state = [con_state; dis_state];
else
    con_state = state(1:model.dsc,1);
    dis_state = state(model.dsc+1:model.ds,1);
end

% Calculate probability if required
if nargout>1
    dis_prob = sum(log(binopdf(dis_state,1,[model.pprior_pres*ones(model.M,1); model.pprior_noise])));
    if any(con_state(1:M)<model.Amin)||any(con_state(M+1:2*M)<model.wmin)
        con_prob = -inf;
    else
        con_prob = loggausspdf(con_state, model.x1_lin_mn, model.x1_lin_vr);
    end
    prob = dis_prob + con_prob;
else
    prob = [];
end

end

