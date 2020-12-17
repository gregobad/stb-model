## Set your local directory
using Distributed
nprocs = 6
addprocs(nprocs)
@everywhere local_dir = "/home/gregorymartin/Dropbox/STBNews"

# @everywhere days_to_use = collect(1:172)
@everywhere days_to_use = [2,7,12,17,22,27,32,36,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170] # every Tuesday
# @everywhere days_to_use = [7,17,27,36,50,60,70,80,90,100,110,120,130,140,150,160,170] # every other Tuesday
@everywhere B = 5  # num path sims

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
@everywhere cd(output_dir)
@everywhere par_init = CSV.File("MCMC_chain1_20days.csv") |> DataFrame;
@everywhere cd(code_dir)
@everywhere include("read_par_from_mcmc.jl")

# # to read direct from csv:
# @everywhere cd(data_dir)
# @everywhere par_init = CSV.File("par_init_20days_restart.csv") |> DataFrame;

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
CSV.write("MCMC_chain1_20days.csv", chain1)

# to produce output for standard plots
ev1 = MA.chains[1].evals[maximum(MA.chains[1].best_id)]
ev1 = stb_obj(ev1; dt=stbdat, save_output=true, store_moments = true)

# to output moments
simmoments = SMM.check_moments(ev1)
select!(simmoments, [:moment, :data, :data_sd, :simulation, :distance])
simmoments.sq_diff = simmoments.distance.^2 .* simmoments.data_sd
sort!(simmoments, :sq_diff)
CSV.write("moments.csv", simmoments)





## grid plots

initial_beta_vote = SMM.paramd(SMM.Eval(mprob))[Symbol("beta:vote")];
initial_horse_race = SMM.paramd(SMM.Eval(mprob))[Symbol("topic_leisure:horse_race")];


bv_values = initial_beta_vote * collect(range(0.8,1.2,length=101))
hr_values = initial_horse_race * collect(range(0.8,1.2,length=101))

bv_fn_grid = zeros(Float64, 101)
bv_mom_grid = zeros(Float64, 101, length(SMM.dataMomentW(SMM.Eval(mprob))))

hr_fn_grid = zeros(Float64, 101)
hr_mom_grid = zeros(Float64, 101, length(SMM.dataMomentW(SMM.Eval(mprob))))

# fill beta_vote
println("beta:vote")
for i in 1:length(bv_values)
    if (mod(i,10) == 0)
        println(i)
    end

    # wrap param vector in OrderedDict
    pb_val = DataStructures.OrderedDict(k => k == "beta:vote" ? bv_values[i] : pb[k].value for k in keys(pb))

    # wrap in Eval object
    ev = SMM.Eval(mprob, pb_val)

    # evaluate
    ev = stb_obj(ev; dt=stbdat, store_moments = true)

    # store moments
    simmoments = SMM.check_moments(ev)
    simmoments.sq_diff = simmoments.distance.^2 .* simmoments.data_sd
    bv_mom_grid[i,:] = simmoments.sq_diff

    # store fval
    bv_fn_grid[i] = ev.value
end

# fill horse_race
println("topic_leisure:horse_race")
for i in 1:length(hr_values)
    if (mod(i,10) == 0)
        println(i)
    end
    # wrap param vector in OrderedDict
    pb_val = DataStructures.OrderedDict(k => k == "topic_leisure:horse_race" ? hr_values[i] : pb[k].value for k in keys(pb))

    # wrap in Eval object
    ev = SMM.Eval(mprob, pb_val)

    # evaluate
    ev = stb_obj(ev; dt=stbdat, store_moments = true)

    # store moments
    simmoments = SMM.check_moments(ev)
    simmoments.sq_diff = simmoments.distance.^2 .* simmoments.data_sd
    hr_mom_grid[i,:] = simmoments.sq_diff

    # store fval
    hr_fn_grid[i] = ev.value
end

using Plots
Plots.plot(bv_fn_grid)

overall_mean = dropdims(mapslices(Statistics.mean, bv_mom_grid;dims=1); dims=1)

sortperm(overall_mean)

CSV.write("bv_moments_grid.csv", Tables.table(bv_mom_grid))
CSV.write("hr_moments_grid.csv", Tables.table(hr_mom_grid))

overall_var = dropdims(mapslices(Statistics.var, bv_mom_grid;dims=1); dims=1)

sortperm(overall_var)
