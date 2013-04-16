function algo = lg_setalgo(test, model, al)

% Algorithm parameters

% Number of particles
algo.N = test.num_filt_pts(al);

% Stochastic smooth update?
algo.flag_stochastic = true;
if algo.flag_stochastic
    algo.Dscale = 1;
else
    algo.Dscale = 0;
end

end