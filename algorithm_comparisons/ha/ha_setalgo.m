function algo = ha_setalgo(test, model, al)

% Algorithm parameters

% Number of particles
algo.N = test.num_filt_pts(al);

% Stochastic smooth update?
algo.flag_stochastic = false;
if algo.flag_stochastic
    algo.D = 1E-6*eye(model.ds);
else
    algo.D = zeros(model.ds);
end

end