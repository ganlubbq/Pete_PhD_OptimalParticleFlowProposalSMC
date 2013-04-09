function algo = lg_setalgo(test, model, al)

% Algorithm parameters

% Number of particles
algo.N = test.num_filt_pts(al);

% Stochastic smooth update?
algo.flag_stochastic = false;
if algo.flag_stochastic
    algo.D = blkdiag(1E-2*eye(2), 1E-3*eye(2));
else
    algo.D = zeros(model.ds);
end

end