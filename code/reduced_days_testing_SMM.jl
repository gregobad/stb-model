## Set your local directory

using Distributed
nprocs = 2
addprocs(nprocs)
@everywhere local_dir = "/home/gregorymartin/Dropbox/STBNews"

### END OPTIONS ###

## Directory locations for code and data
@everywhere using Printf
@everywhere code_dir = @sprintf("%s/stb-model/code/", local_dir)
@everywhere data_dir = @sprintf("%s/stb-model/data", local_dir)
@everywhere output_dir = @sprintf("%s/stb-model/output", local_dir)

### LOAD OBJECTIVE AND DATA ###
## parallel version ##
@everywhere cd(code_dir)
# include("load_model_data.jl")
@everywhere include("load_model_data_limited_days.jl")


## setup for interim output


@everywhere par_init = CSV.read("par_init.csv");
@everywhere par_init = join(par_init, par_init_og, on=:par, kind=:semi)

# change parameters here if desired
@everywhere to_optimize = String.(par_init.par)
@everywhere lb = Float64.(par_init.lb)
@everywhere ub = Float64.(par_init.ub)
@everywhere x3 = [0.0856111, 0.298816, 0.672018, 0.00791401, 0.0172097, 0.0403852, 0.0182825, 0.542866, -8.58113, -7.96851, 5.74632, 9.03305, 3.69841, 3.06432, 16.8124, -3.67009, 3.93393, -0.304003, -6.11472, -3.02162, -6.32727, -11.7381, -4.55283, -4.99359, 6.06492, 3.60548, 3.16662, 0.718529, 9.68126, 0.279528, 0.267072, -0.769708, -0.583047, -0.355038, -0.633251, -0.337484, -1.00363, -1.01607, 1.12577, -0.571604, -0.196189, -0.391476, 0.712405, 0.177007, 1.75119, 0.621433, 1.38874, -0.804487, 0.500555, 0.947754, -0.290571, 0.19874, 0.567433, -1.5762, -2.39216, -0.163784, -1.56854, 1.17207, 0.873464, 1.83316, -0.00838412, 1.14025, -0.540319, -0.639725, 0.264277, 0.208561, 1.40747, 0.340672]

@everywhere x3[1:4] = x3[1:4] ./ sum(x3[1:4])

@everywhere for k in to_optimize
    i = findfirst(k .== to_optimize)
    SMM.addSampledParam!(mprob, k, x3[i], lb[i], ub[i])
end


opts = Dict(
    "N" => nprocs,
    # "maxiter" => 5,
    "maxiter"=>1000,
    "maxtemp" => nprocs,
    "sigma" => 0.005,
    "sigma_update_steps" => 10,
    "sigma_adjust_by" => 0.05,
    "smpl_iters" => 1000,
    "parallel" => true,
    "min_improve" => [0.0 for i = 1:nprocs],
    "acc_tuners" => [1.0 for i = 1:nprocs],
    "animate" => false,
    "batch_size" => 8,
    "single_par_draw" => true     # set true to enable one-parameter-at-a-time jumps
    # "save_frequency"=>100
    # "filename"=>"MCMC_chain_state.jld2"
    # dist_fun => -
);

@everywhere cd(output_dir)

## set-up BGP algorithm (modified version that respects lambdas summing to 1 constraint):
MA = MAlgoSTB(mprob, opts);
wp = CachingPool(workers());

### MAIN MCMC LOOP ###
SMM.run!(MA);

summary(MA)
chain1 = history(MA.chains[1]);

# to produce output for standard plots
stb_obj(ev; dt=stbdat, save_output = true)


# to produce moments
out = stb_obj(ev; dt=stbdat, store_moments = true)
