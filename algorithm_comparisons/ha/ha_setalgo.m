function algo = ha_setalgo(test, model, al)

% Algorithm parameters

% Number of particles
algo.N = test.num_filt_pts(al);

% Intermediate resampling
algo.flag_intermediate_resample = false;

% Stochastic smooth update?
if test.flag_stochastic
    algo.Dscale = test.Dscale;
else
    algo.Dscale = 0;
end

end