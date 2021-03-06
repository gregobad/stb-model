### OPTIONS: change things here ###
## number of cores to use in parallel
const numprocs = 6;
using Distributed
addprocs(numprocs)

## Set your local directory
@everywhere local_dir = "/home/gregorymartin/Dropbox/STBNews"

### END OPTIONS ###

## Directory locations for code and data
@everywhere using Printf
@everywhere code_dir = @sprintf("%s/stbmodel/code/", local_dir)
@everywhere data_dir = @sprintf("%s/stbmodel/data", local_dir)
@everywhere output_dir = @sprintf("%s/stbmodel/output", local_dir)

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

# store as initial parameter vector in mprob object
@everywhere SMM.addSampledParam!(mprob,pb);

## define subset of pars to optimize over
@everywhere to_optimize = String.(par_init.par[findall(occursin.(r"info|slant|topic_mu|topic_leisure", par_init.par))])
@everywhere ub = par_init[findall(occursin.(r"info|slant|topic_mu|topic_leisure", par_init.par)), :ub]
@everywhere lb = par_init[findall(occursin.(r"info|slant|topic_mu|topic_leisure", par_init.par)), :lb]
# @everywhere to_optimize = String.(par_init.par)
@everywhere x0 = [pb[k].value for k in to_optimize]

# ## grid search

# x0_save = copy(x0)
# f = 0.0
#
# @everywhere using Plots
#
#
# grid = @distributed (hcat) for p in to_optimize
#     println(p)
#     x0 .= x0_save
#     i = findfirst(p .== to_optimize)
#     beg_seq = min((x0[i] * 0.8), (x0[i] * 1.2))
#     end_seq = max((x0[i] * 0.8), (x0[i] * 1.2))
#     step = (end_seq - beg_seq) / 50
#     xs = beg_seq:step:end_seq
#     fs = [fgh!(f, nothing, nothing, setindex!(x0, x, i)) for x in xs]
#     png(plot(xs, fs), "/home/gregorymartin/Dropbox/STBNews/data/model/grid_plots/" * p)
#     fs
# end
#
#
# CSV.write(file="/home/gregorymartin/Dropbox/STBNews/data/model/grid_plots/grid_search.csv", DataFrame(grid))



@everywhere function f0(x)
    # wrap param vector in OrderedDict
    pb_val = DataStructures.OrderedDict(k => k ∈ to_optimize ? x[findfirst(k .== to_optimize)] : pb[k].value for k in keys(pb))

    # wrap in Eval object
    ev = SMM.Eval(mprob, pb_val)

    stb_obj(ev; dt=stbdat).value
end

@everywhere import BlackBoxOptim
@everywhere init_pop = [x0 BlackBoxOptim.rand_individuals(BlackBoxOptim.ContinuousRectSearchSpace(lb, ub),23;method = :latin_hypercube)]
wp = CachingPool(workers())
opt_setup = BlackBoxOptim.bbsetup(f0;
    SearchSpace = BlackBoxOptim.ContinuousRectSearchSpace(lb, ub),
    Population = init_pop,
    MaxFuncEvals = 4000,
    TraceInterval = 60.0,
    TraceMode = :verbose,
    Workers = workers())

optimized = BlackBoxOptim.run!(opt_setup)

println(BlackBoxOptim.best_candidate(optimized))
println(BlackBoxOptim.best_fitness(optimized))

### TESTING ZONE ###
# # to evaluate at start point
# obj = stb_obj(SMM.Eval(mprob); dt=stbdat)
