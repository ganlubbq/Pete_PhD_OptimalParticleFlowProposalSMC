function algo = ha_setalgo(test, model, al)

% Algorithm parameters

% Number of particles
algo.N = test.num_filt_pts(al);

% Intermediate resampling
algo.flag_intermediate_resample = true;

% Stochastic smooth update?
algo.flag_stochastic = true;
if algo.flag_stochastic
    algo.D = 1E-5*eye(model.ds);
else
    algo.D = zeros(model.ds);
end

end