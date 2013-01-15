function algo = linearGaussian_set_algo

% Algorithm parameters

algo.N = 100;                   % Number of particles in PF
algo.L = 10;                   % Number of integration steps in particle flow proposal
algo.M = 20;                     % Number of MCMC steps in MH proposal

end