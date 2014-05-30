function algo = drone_setalgo(test, model, al)

% Algorithm parameters

% Number of particles
algo.N = test.num_filt_pts(al);

% Stochastic smooth update?
algo.flag_intermediate_resample = test.flag_intermediate_resample;

if ((al==6)||(al==5))&&(test.flag_stochastic)
    algo.Dscale = test.Dscale;
else
    algo.Dscale = 0;
end

end