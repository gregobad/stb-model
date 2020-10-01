cd(data_dir)

par_init = CSV.read("MCMC_chain1.csv");
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

# norm innovations to sd 1
par_init.value[findall(occursin.(r"_t01", par_init.par))] = par_init.value[findall(occursin.(r"_t01", par_init.par))] ./ Statistics.std(par_init.value[findall(occursin.(r"_t01", par_init.par))])
par_init.value[findall(occursin.(r"_t02", par_init.par))] = par_init.value[findall(occursin.(r"_t02", par_init.par))] ./ Statistics.std(par_init.value[findall(occursin.(r"_t02", par_init.par))])
par_init.value[findall(occursin.(r"_t03", par_init.par))] = par_init.value[findall(occursin.(r"_t03", par_init.par))] ./ Statistics.std(par_init.value[findall(occursin.(r"_t03", par_init.par))])
par_init.value[findall(occursin.(r"_t04", par_init.par))] = par_init.value[findall(occursin.(r"_t04", par_init.par))] ./ Statistics.std(par_init.value[findall(occursin.(r"_t04", par_init.par))])

par_init = join(par_init, par_init_og, on = :par, kind=:right);
par_init.value = coalesce.(par_init.value, 0.0)

# the following 2 lines zero out the tz-hour and show dummies, respectively
filter!(row -> !occursin(r"beta:h\d", row[:par]), par_init)
# filter!(row -> !occursin("beta:show", row[:par]), par_init)

## dictionary indexed by parameter, with initial values and bounds for each
pb = DataStructures.OrderedDict(zip(
    par_init.par,
    DataFrames.eachrow(par_init[:, [:value, :lb, :ub]]),
));
