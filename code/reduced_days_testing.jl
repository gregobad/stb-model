## Set your local directory
local_dir = "/home/gregorymartin/Dropbox/STBNews"

### END OPTIONS ###

## Directory locations for code and data
using Printf
code_dir = @sprintf("%s/stb-model/code/", local_dir)
data_dir = @sprintf("%s/stb-model/data", local_dir)
output_dir = @sprintf("%s/stb-model/output", local_dir)

### LOAD OBJECTIVE AND DATA ###
## parallel version ##
cd(code_dir)
# include("load_model_data.jl")
include("load_model_data_limited_days.jl")


### READ PARAMETER VECTOR ###
# to read from last MCMC run:
# cd(code_dir)
# include("read_par_from_mcmc.jl")

# to read direct from csv:
par_init = CSV.read("par_init.csv");
par_init = join(par_init, par_init_og, on=:par, kind=:semi)

# dictionary indexed by parameter, with initial values and bounds for each
pb = DataStructures.OrderedDict(zip(
    par_init.par,
    DataFrames.eachrow(par_init[:, [:value, :lb, :ub]]),
));


# add pars to mprob object
SMM.addSampledParam!(mprob,pb);

# change parameters here if desired
to_optimize = String.(par_init.par)
x0 = par_init.value
pb_val = DataStructures.OrderedDict(k => k ∈ to_optimize ? x0[findfirst(k .== to_optimize)] : pb[k].value for k in keys(pb))
ub = par_init.ub
lb = par_init.lb

# wrap in Eval object
ev = SMM.Eval(mprob, pb_val)

cd(output_dir)

# examine moments
result = stb_obj(ev; dt=stbdat, store_moments=true)

moment_compare = SMM.check_moments(result)
moment_compare[moment_compare.moment .== :FNC_MSN_pct_0005,:]


## BlackBoxOptim setup
import BlackBoxOptim

function f0(x)
    # wrap param vector in OrderedDict
    pb_val = DataStructures.OrderedDict(k => k ∈ to_optimize ? x[findfirst(k .== to_optimize)] : pb[k].value for k in keys(pb))

    # wrap in Eval object
    ev = SMM.Eval(mprob, pb_val)

    stb_obj(ev; dt=stbdat).value
end

# initial population: best previously found value plus other randomly generated ones nearby
random_perturb = 0.2.*Random.rand(rng, length(x0), 24) .+ 0.9;
perturbed = x0 .* random_perturb
perturbed = min.(perturbed, ub)
perturbed = max.(perturbed, lb)
perturbed[1:4,:] = perturbed[1:4,:] ./ sum(perturbed[1:4,:],dims=1)

init_pop = [x0 perturbed]

# init_pop = [x0 BlackBoxOptim.rand_individuals(BlackBoxOptim.ContinuousRectSearchSpace(lb, ub),23;method = :latin_hypercube)]

## setup for interim output
cd(output_dir)
outrow = Dict(Symbol(to_optimize[i]) => x0[i] for i = 1:length(to_optimize))
outrow[Symbol("n_func_evals")] = 0
outrow[Symbol("fval")] = f0(x0)
output_df = DataFrames.DataFrame(;outrow...)
CSV.write("bboptim_progress.csv", output_df)

function callback(oc)
    # save output each step
    x1 = BlackBoxOptim.best_candidate(oc)

    outrow = Dict(Symbol(to_optimize[i]) => x1[i] for i = 1:length(to_optimize))
    outrow[Symbol("n_func_evals")] = BlackBoxOptim.num_func_evals(oc)
    outrow[Symbol("fval")] = BlackBoxOptim.best_fitness(oc)

    output_df = DataFrames.DataFrame(;outrow...)
    CSV.write("bboptim_progress.csv", output_df; append=true)
end

opt_setup = BlackBoxOptim.bbsetup(f0;
    SearchSpace = BlackBoxOptim.ContinuousRectSearchSpace(lb, ub),
    Population = poop,
    MaxFuncEvals = 12000,
    TraceInterval = 60.0,
    TraceMode = :verbose,
    CallbackFunction = callback,
    CallbackInterval = 60.0)

optimized = BlackBoxOptim.run!(opt_setup)


# to produce output for standard plots
stb_obj(ev; dt=stbdat, save_output = true)


# to produce moments
out = stb_obj(ev; dt=stbdat, store_moments = true)
