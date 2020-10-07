using Distributed
@everywhere import BlackBoxOptim

## Set your local directory
@everywhere local_dir = "/home/groups/gjmartin"

## Directory locations for code and data
@everywhere using Printf
@everywhere code_dir = @sprintf("%s/stb-model/code/", local_dir)
@everywhere data_dir = @sprintf("%s/stb-model/data", local_dir)
@everywhere output_dir = @sprintf("%s/stb-model/output", local_dir)

### LOAD OBJECTIVE AND DATA ###
## parallel version ##
@everywhere cd(code_dir)
@everywhere include("load_model_data.jl")


### SETUP INITIAL PARAMETER VECTOR ###
# to read from last MCMC run:
# @everywhere cd(code_dir)
# @everywhere include("read_par_from_mcmc.jl")

# to read direct from csv:
@everywhere par_init = CSV.read("par_init.csv");

# dictionary indexed by parameter, with initial values and bounds for each
@everywhere pb = DataStructures.OrderedDict(zip(
    par_init.par,
    DataFrames.eachrow(par_init[:, [:value, :lb, :ub]]),
));


### DIRECT OPTIMIZATION

## store as initial parameter vector in mprob object
@everywhere SMM.addSampledParam!(mprob,pb);

## define subset of pars to optimize over
# @everywhere to_optimize = String.(par_init.par[findall(occursin.(r"info|slant|topic_mu|topic_leisure", par_init.par))])
# @everywhere ub = par_init[findall(occursin.(r"info|slant|topic_mu|topic_leisure", par_init.par)), :ub]
# @everywhere lb = par_init[findall(occursin.(r"info|slant|topic_mu|topic_leisure", par_init.par)), :lb]
## or do everything simultaneously
@everywhere to_optimize = String.(par_init.par)
@everywhere ub = par_init.ub
@everywhere lb = par_init.lb

## include initial parameter vector in the population
@everywhere x0 = [pb[k].value for k in to_optimize]

@everywhere cd(output_dir)

## the function to optimize
@everywhere function f0(x)
    # wrap param vector in OrderedDict
    pb_val = DataStructures.OrderedDict(k => k ∈ to_optimize ? x[findfirst(k .== to_optimize)] : pb[k].value for k in keys(pb))

    # wrap in Eval object
    ev = SMM.Eval(mprob, pb_val)

    stb_obj(ev; dt=stbdat).value
end

## initial population: best previously found value plus other randomly generated ones
@everywhere init_pop = [x0 BlackBoxOptim.rand_individuals(BlackBoxOptim.ContinuousRectSearchSpace(lb, ub),49;method = :latin_hypercube)]

## setup for interim output
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
    Population = init_pop,
    MaxFuncEvals = 4000,
    TraceInterval = 60.0,
    TraceMode = :silent,
    Workers = workers(),
    CallbackFunction = callback,
    CallbackInterval = 60.0)

optimized = BlackBoxOptim.run!(opt_setup)

println(BlackBoxOptim.best_candidate(optimized))
println(BlackBoxOptim.best_fitness(optimized))






### TESTING ZONE ###
# # to evaluate at start point
# obj = stb_obj(SMM.Eval(mprob); dt=stbdat)
