# include following only if reading most recent results on server (MCMC_chain1.csv)
# par_init.accepted = par_init.accepted .== "TRUE" # reading in as string, not logical
par_init = par_init[(par_init.accepted.==1),:];
best_draw = argmin(par_init.curr_val);
par_init_val = collect(Float64, par_init[best_draw,8:size(par_init, 2)]);
par_init_par = string.(names(par_init[:,8:size(par_init, 2)]));
 par_init = DataFrames.DataFrame(
    par = par_init_par,
    value = par_init_val
);
# end of run MCMC_chain1-specific code

## dictionary indexed by parameter, with initial values and bounds for each
pb_val = DataStructures.OrderedDict(zip(
    par_init.par,
    par_init.value
))
