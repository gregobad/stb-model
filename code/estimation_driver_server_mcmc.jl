### OPTIONS: change things here ###

## number of iterations for each MCMC chain
const niter = 1200;

## number of cores to use in parallel
const numprocs = 25;

using Distributed
addprocs(numprocs)

## number of chains to run
nchains = numprocs;

## Set your local directory
# @everywhere local_dir = "/Users/mlinegar/Dropbox/STBNews"
@everywhere local_dir = "/home/cfuser/mlinegar"
### END OPTIONS ###

## Directory locations for code and data
@everywhere using Printf
@everywhere code_dir = @sprintf("%s/code/model/julia/", local_dir)
@everywhere data_dir = @sprintf("%s/data/model/", local_dir)

### LOAD OBJECTIVE AND DATA ###
## parallel version ##
@everywhere cd(code_dir)
@everywhere include("load_model_data.jl")


### SETUP INITIAL PARAMETER VECTOR ###
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
# par_init.value[findall(occursin.(r"_t01", par_init.par))] = par_init.value[findall(occursin.(r"_t01", par_init.par))] ./ Statistics.std(par_init.value[findall(occursin.(r"_t01", par_init.par))])
# par_init.value[findall(occursin.(r"_t02", par_init.par))] = par_init.value[findall(occursin.(r"_t02", par_init.par))] ./ Statistics.std(par_init.value[findall(occursin.(r"_t02", par_init.par))])
# par_init.value[findall(occursin.(r"_t03", par_init.par))] = par_init.value[findall(occursin.(r"_t03", par_init.par))] ./ Statistics.std(par_init.value[findall(occursin.(r"_t03", par_init.par))])
# par_init.value[findall(occursin.(r"_t04", par_init.par))] = par_init.value[findall(occursin.(r"_t04", par_init.par))] ./ Statistics.std(par_init.value[findall(occursin.(r"_t04", par_init.par))])
#
par_init = join(par_init, par_init_og, on = :par, kind=:right);
par_init.value = coalesce.(par_init.value, 0.0)
#
# # the following 2 lines zero out the tz-hour and show dummies, respectively
# filter!(row -> !occursin(r"beta:h\d", row[:par]), par_init)
# filter!(row -> !occursin("beta:show", row[:par]), par_init)


# reset beta_channel_mu and sigma to ensure negative channel_mu, standardize mean to 1
# # match previous variance
#
# ln_mus = par_init.value[occursin.("beta:channel_mu", par_init.par)]
# ln_sigs = par_init.value[occursin.("beta:channel_sigma", par_init.par)]
# vars  = (exp.(ln_sigs.^2) .- 1) .* exp.(2 .* ln_mus .+ ln_sigs.^2)
# means = exp.(ln_mus .+ (ln_sigs.^2)./2)
#
# par_init.value[occursin.("beta:show:", par_init.par)] .+= (means .- 1)
# par_init.value[occursin.("beta:channel_sigma", par_init.par)] = sqrt.(log.(1 .+ vars))
# par_init.value[occursin.("beta:channel_mu", par_init.par)] = -(log.(1 .+ vars))/2
#

## dictionary indexed by parameter, with initial values and bounds for each
pb = DataStructures.OrderedDict(zip(
    par_init.par,
    DataFrames.eachrow(par_init[:, [:value, :lb, :ub]]),
));

# # run on full set of parameters
SMM.addSampledParam!(mprob,pb);

## estimation options:
opts = Dict(
    "N" => nchains,
    # "maxiter" => 5,
    "maxiter"=>niter,
    "maxtemp" => numprocs,
    "sigma" => 0.005,
    "sigma_update_steps" => 10,
    "sigma_adjust_by" => 0.05,
    "smpl_iters" => 1000,
    "parallel" => true,
    "min_improve" => [0.0 for i = 1:nchains],
    "acc_tuners" => [1.0 for i = 1:nchains],
    "animate" => false,
    "batch_size" => 8,
    "single_par_draw" => true     # set true to enable one-parameter-at-a-time jumps
    # "save_frequency"=>100
    # "filename"=>"MCMC_chain_state.jld2"
    # dist_fun => -
);

## set-up BGP algorithm (modified version that respects lambdas summing to 1 constraint):
MA = MAlgoSTB(mprob, opts);
wp = CachingPool(workers());

### MAIN MCMC LOOP ###
SMM.run!(MA);


### POST PROCESSING OF RESULTS ###
summary(MA)
chain1 = history(MA.chains[1]);

CSV.write("MCMC_chain1.csv", chain1)
var = [MA.chains[1].sigma]
var = DataFrame(var = var)
CSV.write("MCMC_variance.csv", var, writeheader=false)

# save to produce std output
obj = stb_obj(MA.chains[1].evals[maximum(MA.chains[1].best_id)]; dt=stbdat, save_output=true, store_moments=true)





# ### TESTING ZONE ###
# # to evaluate at start point
# obj = stb_obj(SMM.Eval(mprob); dt=stbdat)
#
# # to evaluate at best MCMC chain state
# obj = stb_obj(MA.chains[1].evals[maximum(MA.chains[1].best_id)]; dt=stbdat)
# # MA.chains[1].evals[].simMoments
# # MA.chains[1].evals[].dataMoments
# # datamoments = SMM.dataMomentd(MA.chains[1].evals[])
# # simmoments = SMM.check_moments(MA.chains[1].evals[])
# simmoments = SMM.check_moments(obj)
# CSV.write("julia_MCMC_moments.csv", simmoments)
# # CSV.write("correct_draw_order_moments.csv", simmoments)
# # CSV.write("incorrect_draw_order_moments.csv", simmoments)
