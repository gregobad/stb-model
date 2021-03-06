using Distributed
nprocs = 6
addprocs(nprocs)
@everywhere local_dir = "C:/Dropbox (gsc)/STBNews"

@everywhere days_to_use = collect(1:172)
# @everywhere days_to_use = [7,17,27,36,50,60,70,80,90,100,110,120,130,140,150,160,170] # every other Tuesday
@everywhere B = 10  # num path sims

# to sample only main parameters
@everywhere tree_base = "main"

# # to sample only path parameters
# @everywhere tree_base = "path"

# # to sample all
# @everywhere tree_base = ""

### END OPTIONS ###

## Directory locations for code and data
@everywhere using Printf
@everywhere code_dir = @sprintf("%s/stb-model-discrete/code/", local_dir)
@everywhere data_dir = @sprintf("%s/stb-model-discrete/data", local_dir)
@everywhere output_dir = @sprintf("%s/stb-model-discrete/output", local_dir)
@everywhere sampling_dir = @sprintf("%s/stb-model-discrete/sampling", local_dir)

## LOAD OBJECTIVE AND DATA ##
@everywhere cd(code_dir)
@everywhere include("load_model_data.jl")

## READ PARAMETER VECTOR ##
# to read from last MCMC run:
@everywhere cd(code_dir)
@everywhere include("read_par_from_mcmc.jl")

# # to read direct from csv:
# @everywhere cd(data_dir)
# @everywhere par_init = CSV.File("par_init.csv") |> DataFrame;

# merge with the bounds definition
@everywhere par_init = innerjoin(par_init[:,[:par,:value]], par_init_og, on=:par)

# create dictionary indexed by parameter, with initial values and bounds for each
@everywhere pb = DataStructures.OrderedDict(zip(
    par_init.par,
    DataFrames.eachrow(par_init[:, [:value, :lb, :ub]]),
));

# add pars to mprob object
SMM.addSampledParam!(mprob,pb);

# options for MCMC chain
opts = Dict(
    "N" => nprocs,
    "maxiter"=>5,
    "maxtemp" => nprocs/2,
    "sigma" => 0.005,
    "sigma_update_steps" => 100,
    "sigma_adjust_by" => 0.05,
    "smpl_iters" => 1000,
    "parallel" => true,
    "min_improve" => [0.0 for i = 1:nprocs],
    "acc_tuners" => [1.0 for i = 1:nprocs],
    "animate" => false,
    "sampling_scheme" => sample_tree
);


@everywhere cd(output_dir)

## set-up BGP algorithm and worker pool to run objective evaluations
MA = MAlgoSTB(mprob, opts)
wp = CachingPool(workers())

### MAIN MCMC LOOP ###
SMM.run!(MA);

summary(MA)
chain1 = history(MA.chains[1]);
CSV.write("MCMC_chain1.csv", chain1)

# to produce output for standard plots
ev1 = MA.chains[1].evals[maximum(MA.chains[1].best_id)]
ev1 = stb_obj(ev1; dt=stbdat, save_output=true, store_moments = true)

# to output moments
simmoments = SMM.check_moments(ev1)
select!(simmoments, [:moment, :data, :data_sd, :simulation, :distance])
simmoments.sq_diff = simmoments.distance.^2 .* simmoments.data_sd
sort!(simmoments, :sq_diff)
CSV.write("moments.csv", simmoments)



## restarting from best parameter found
include("read_par_from_mcmc.jl")
for (k,v) in pb_val
    mprob.initial_value[k] = v
end

MA = MAlgoSTB(mprob, opts);
wp = CachingPool(workers());

### MAIN MCMC LOOP ###
SMM.run!(MA);

summary(MA)
chain1 = history(MA.chains[1]);
CSV.write("MCMC_chain1.csv", chain1)


restart!(MA, 3000)
