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
par_init = CSV.read("par_init.csv", DataFrame);
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

## BlackBoxOptim setup
import BlackBoxOptim

function f0(x)
    # wrap param vector in OrderedDict
    pb_val = DataStructures.OrderedDict(k => k ∈ to_optimize ? x[findfirst(k .== to_optimize)] : pb[k].value for k in keys(pb))

    # wrap in Eval object
    ev = SMM.Eval(mprob, pb_val)

    stb_obj(ev; dt=stbdat).value
end

# initial population: best previously found value plus other randomly generated ones
init_pop = [x0 BlackBoxOptim.rand_individuals(BlackBoxOptim.ContinuousRectSearchSpace(lb, ub),23;method = :latin_hypercube)]

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
    Method = :generating_set_search,
    Population = init_pop,
    MaxFuncEvals = 4000,
    TraceInterval = 60.0,
    TraceMode = :verbose,
    CallbackFunction = callback,
    CallbackInterval = 60.0)

optimized = BlackBoxOptim.run!(opt_setup)


# to produce output for standard plots
stb_obj(ev; dt=stbdat, save_output = true)


# to produce moments
out = stb_obj(ev; dt=stbdat, store_moments = true)

## best candidate from run on 10/9
x1 = [0.0538291, 0.310566, 0.610029, 0.0027239, 0.0117096, 0.0422994, 0.0179162, 0.284606, -8.64689, -7.07916, 5.78637, 9.14795, 7.08874, 3.78098, 17.4236, -3.71464, 6.56188, -0.303443, -5.9517, -2.51041, -5.88717, -12.1327, -4.75287, -5.75435, 6.22501, 3.68311, 3.50384, 0.721689, 8.72172, 0.337804, 0.220523, -1.18129, -0.738839, -0.478915, -1.1311, -0.0321939, -0.428803, -0.875633, 1.42455, -0.599551, 1.11229, -1.01874, -0.230752, 0.350092, -0.39229, 0.636314, 1.40058, 0.431968, 0.150146, 0.870525, -0.343516, 0.384235, 0.0727912, -0.582744, -2.1689, -0.236028, 1.03625, 1.09544, 0.47114, 1.79075, 0.000996366, 0.811904, -0.439734, -0.270028, 0.527055, 0.178262, 1.26242, 0.460383]
pb_val = DataStructures.OrderedDict(k => k ∈ to_optimize ? x1[findfirst(k .== to_optimize)] : pb[k].value for k in keys(pb))

# wrap in Eval object
ev = SMM.Eval(mprob, pb_val)


# examine moments
result = stb_obj(ev; dt=stbdat, store_moments=true, save_output=true)

moment_compare = SMM.check_moments(result)
moment_compare[moment_compare.moment .== :CNN_FNC_pct_001,:]
