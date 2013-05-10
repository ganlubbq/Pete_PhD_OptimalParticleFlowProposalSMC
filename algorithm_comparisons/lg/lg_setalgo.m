function algo = lg_setalgo(test, model, al)

% Algorithm parameters

% Number of particles
algo.N = test.num_filt_pts(al);

% Stochastic smooth update?
if (al==5) && (test.flag_stochastic)
    algo.Dscale = test.Dscale;
else
    algo.Dscale = 0;
end

end