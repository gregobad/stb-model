## Set your local directory
local_dir = "/home/gregorymartin/Dropbox/STBNews"

### END OPTIONS ###

## Directory locations for code and data
using Printf
code_dir = @sprintf("%s/stb-model-discrete/code/", local_dir)
data_dir = @sprintf("%s/stb-model-discrete/data", local_dir)
output_dir = @sprintf("%s/stb-model-discrete/output", local_dir)

### LOAD OBJECTIVE AND DATA ###
## parallel version ##
cd(code_dir)
include("load_model_data.jl")


### READ PARAMETER VECTOR ###
# to read from last MCMC run:
# cd(code_dir)
# include("read_par_from_mcmc.jl")

# to read direct from csv:
par_init = CSV.File("par_init.csv") |> DataFrame;
par_init = innerjoin(par_init[:,[:par,:value]], par_init_og, on=:par)

# dictionary indexed by parameter, with initial values and bounds for each
pb = DataStructures.OrderedDict(zip(
    par_init.par,
    DataFrames.eachrow(par_init[:, [:value, :lb, :ub]]),
));


# add pars to mprob object
SMM.addSampledParam!(mprob,pb);

ev = SMM.Eval(mprob)

result = stb_obj(ev; dt=stbdat)

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
result = stb_obj(ev; dt=stbdat)

moment_compare = SMM.check_moments(result)
moment_compare[moment_compare.moment .== :CNN_FNC_pct_0005,:]


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

random = BlackBoxOptim.rand_individuals(BlackBoxOptim.ContinuousRectSearchSpace(lb, ub),25;method = :latin_hypercube)
random[1:4,:] = random[1:4,:] ./ sum(random[1:4,:],dims=1)

init_pop = [x0 perturbed random]

# init_pop = [x0 BlackBoxOptim.rand_individuals(BlackBoxOptim.ContinuousRectSearchSpace(lb, ub),23;method = :latin_hypercube)]

## setup for interim output
cd(output_dir)
outrow = Dict(Symbol(to_optimize[i]) => x0[i] for i = 1:length(to_optimize))
outrow[Symbol("n_func_evals")] = 0
outrow[Symbol("fval")] = f0(x0)
output_df = DataFrames.DataFrame(;outrow...)
CSV.write("bboptim_progress_heterogeneous_topics.csv", output_df)

function callback(oc)
    # save output each step
    x1 = BlackBoxOptim.best_candidate(oc)

    outrow = Dict(Symbol(to_optimize[i]) => x1[i] for i = 1:length(to_optimize))
    outrow[Symbol("n_func_evals")] = BlackBoxOptim.num_func_evals(oc)
    outrow[Symbol("fval")] = BlackBoxOptim.best_fitness(oc)

    output_df = DataFrames.DataFrame(;outrow...)
    CSV.write("bboptim_progress_heterogeneous_topics.csv", output_df; append=true)
end

opt_setup = BlackBoxOptim.bbsetup(f0;
    SearchSpace = BlackBoxOptim.ContinuousRectSearchSpace(lb, ub),
    Population = init_pop,
    MaxFuncEvals = 12000,
    TraceInterval = 60.0,
    TraceMode = :verbose,
    CallbackFunction = callback,
    CallbackInterval = 60.0)

optimized = BlackBoxOptim.run!(opt_setup)

x1 = [0.0851506, 0.0128081, 0.102923, 0.0584577, 7.02271, 1.24777, 7.26563, 6.91289, -0.0746859, 2.93762, 0.0252481, 3.80602, 0.619981, -14.2312, 12.048, -12.816, 2.72627, -1.7952, -19.0591, -9.8325, -19.2587, 0.663264, 2.26935, 5.87609, 0.739916, -3.17489, -1.79067, -0.0338981, -0.482072, 0.550712, 0.469816, -1.00208, 1.68253, 0.269032, -2.96811, 0.79115, -2.03846, -3.29889, 4.18924, 0.7472, 0.556724, -3.09644, -0.320314, 0.473562, -0.192192, -0.758134, 1.2525, -2.2968, -1.72351, -0.50323, -0.102707, 1.52075, 0.559682, 1.17935, 0.310434, 2.14311, 0.041278, -0.521217, -4.34989, -1.36333, 2.98507, 0.513907]

pb_val = DataStructures.OrderedDict(k => k ∈ to_optimize ? x1[findfirst(k .== to_optimize)] : pb[k].value for k in keys(pb))

# to produce output for standard plots
stb_obj(Eval(mprob, pb_val); dt=stbdat, save_output = true)


# to produce moments
out = stb_obj(ev; dt=stbdat, store_moments = true)
